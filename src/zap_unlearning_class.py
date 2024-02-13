"""
This file contains the implementation of Zap Unlearning
Our proposed method
"""

import copy
import os
import time

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from eval import compute_accuracy
from unlearning_base_class import UnlearningBaseClass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SoftCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets):
        log_softmax = F.log_softmax(outputs, dim=1)
        loss = -torch.sum(targets * log_softmax, dim=1)
        return torch.mean(loss)


class ZapUnlearning(UnlearningBaseClass):
    def __init__(self, parent_instance):
        super().__init__(
            parent_instance.dl,
            parent_instance.batch_size,
            parent_instance.num_classes,
            parent_instance.model,
            parent_instance.epochs,
            parent_instance.dataset,
        )
        self.is_multi_label = True if parent_instance.dataset == "mucac" else False
        self.loss_fn = nn.CrossEntropyLoss()
        self.soft_loss_fn = SoftCrossEntropyLoss()
        self.lr = 1e-3
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        self.lr_scheduler = None
        mlflow.log_param("loss", "cross_entropy")
        mlflow.log_param("lr", self.lr)
        mlflow.log_param("momentum", self.momentum)
        mlflow.log_param("weight_decay", self.weight_decay)
        mlflow.log_param("optimizer", self.optimizer)
        mlflow.log_param("lr_scheduler", "None")

    def unlearn_zap_lrp(self, relevance_threshold, set_to_check_relevance):
        mlflow.log_param("relevance_threshold", relevance_threshold)
        mlflow.log_param("set_to_check_relevance", set_to_check_relevance)
        run_time = 0

        # Zap the weights of the weights of the defined neurons.
        if set_to_check_relevance == "forget":
            neuron_contrib = self.lrp_fc_layer(self.dl["forget"])
        elif set_to_check_relevance == "both":
            neuron_contrib_forget = self.lrp_fc_layer(self.dl["forget"])
            neuron_contrib_retain = self.lrp_fc_layer(self.dl["retain"])
            neuron_contrib = neuron_contrib_forget - neuron_contrib_retain
            neuron_contrib = (neuron_contrib - neuron_contrib.min()) / (
                neuron_contrib.max() - neuron_contrib.min()
            )
            neuron_contrib = (neuron_contrib * 2) - 1
        else:
            raise ValueError("set_to_check_relevance must be either 'forget' or 'both'")

        mask_weight, zapped_neurons = self.get_mask_relevant_weights(
            neuron_contrib, threshold=relevance_threshold
        )

        # Forget training (destroy and impair stage)
        for epoch in tqdm(range(self.epochs)):
            start_epoch_time = time.time()

            self._random_init_weights(mask_weight)
            self.model.train()

            # for inputs, targets in self.dl["mock_forget"]:
            for inputs, _ in self.dl["forget"]:
                inputs = inputs.to(DEVICE, non_blocking=True)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                targets = torch.zeros_like(outputs) + 1 / self.num_classes
                targets = targets.to(DEVICE, non_blocking=True)
                loss = self.soft_loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()

            for inputs, targets in self.dl["retain"]:
                inputs = inputs.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)
                # original_logits = original_model(inputs)
                # original_probs = torch.softmax(original_logits, dim=1)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                # loss = self.soft_loss_fn(outputs, original_probs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()

            epoch_run_time = (time.time() - start_epoch_time) / 60  # in minutes
            run_time += epoch_run_time

            acc_retain = compute_accuracy(
                self.model, self.dl["retain"], self.is_multi_label
            )
            acc_forget = compute_accuracy(
                self.model, self.dl["forget"], self.is_multi_label
            )
            acc_val = compute_accuracy(self.model, self.dl["val"], self.is_multi_label)

            # Log accuracies
            mlflow.log_metric("acc_retain", acc_retain, step=(epoch + 1))
            mlflow.log_metric("acc_val", acc_val, step=(epoch + 1))
            mlflow.log_metric("acc_forget", acc_forget, step=(epoch + 1))

        return self.model, run_time, zapped_neurons

    # TODO: Implement this
    def unlearn_zap_fim(self, relevance_threshold, set_to_check_relevance):
        pass

    def unlearn_fim(self):
        run_time = 0
        self._zap_fc_layer()

        # Log accuracies
        log_time_start = time.time()
        acc_retain = compute_accuracy(
            self.model, self.dl["retain"], self.is_multi_label
        )
        acc_forget = compute_accuracy(
            self.model, self.dl["forget"], self.is_multi_label
        )
        acc_val = compute_accuracy(self.model, self.dl["val"], self.is_multi_label)
        mlflow.log_metric("acc_retain", acc_retain, step=0)
        mlflow.log_metric("acc_val", acc_val, step=0)
        mlflow.log_metric("acc_forget", acc_forget, step=0)
        log_time = (time.time() - log_time_start) / 60

        for epoch in tqdm(range(self.epochs)):
            start_time = time.time()
            self.model.train()
            fim_diagonal = self._compute_fim_diagonal()
            self.model.train()
            for inputs, targets in self.dl["retain"]:
                inputs = inputs.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                self.model.zero_grad()
                loss.backward()
                self._newtons_update(fim_diagonal)
            epoch_run_time = (time.time() - start_time) / 60  # in minutes
            run_time += epoch_run_time

            acc_retain = compute_accuracy(
                self.model, self.dl["retain"], self.is_multi_label
            )
            acc_forget = compute_accuracy(
                self.model, self.dl["forget"], self.is_multi_label
            )
            acc_val = compute_accuracy(self.model, self.dl["val"], self.is_multi_label)

            # Log accuracies
            mlflow.log_metric("acc_retain", acc_retain, step=(epoch + 1))
            mlflow.log_metric("acc_val", acc_val, step=(epoch + 1))
            mlflow.log_metric("acc_forget", acc_forget, step=(epoch + 1))

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        return self.model, run_time - log_time

    def _newtons_update(self, fim_diagonal):
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Approximate Hessian diagonal element
                hessian_diag = fim_diagonal[name]

                # Avoid division by zero; add a small epsilon
                epsilon = 1e-5
                hessian_diag += epsilon

                # Newton's update step
                param = param - self.lr * param.grad / hessian_diag

    # Function to compute the diagonal of the Fisher Information Matrix
    def _compute_fim_diagonal(self):
        fim_diagonal = {
            name: torch.zeros_like(param, dtype=torch.float32)
            for name, param in self.model.named_parameters()
        }

        self.model.eval()
        for inputs, targets in self.dl["retain"]:
            inputs = inputs.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)
            # Forward pass
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)

            # Backward pass
            self.model.zero_grad()
            loss.backward()

            # Accumulate squared gradients (Fisher Information Matrix diagonal)
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fim_diagonal[name] += param.grad**2

        # Average over the dataset
        num_data = len(self.dl["retain"])
        for name in fim_diagonal:
            fim_diagonal[name] /= num_data

        return fim_diagonal

    def _random_init_weights(self, weight_mask) -> None:
        """
        This function resets the weights of the last fully connected layer of a neural network.
        If the_class is None, then all the weights of the fc layer are reset.
        If the_class is an integer, then only the weights corresponding to the the_class the_class are reset.
        Args:
            model (torch.nn.Module): The model to unlearn.
            weight_mask (torch.Tensor): It contains ones for the weights to be zapped, zeros for the others.
        """
        fc_layer = self.model.get_last_linear_layer()
        # Get the weights of the fc layer
        weights_reset = fc_layer.weight.data.detach().clone()
        # Reset the weights corresponding the the_class i (fc --> lr)
        torch.nn.init.xavier_normal_(tensor=weights_reset, gain=1.0)
        # torch.nn.init.kaiming_normal_(tensor=weights_reset, mode="fan_out", nonlinearity="relu")
        # torch.nn.init.orthogonal_(tensor=weights_reset, gain=1.0)
        # torch.nn.init.uniform_(tensor=weights_reset, a=-0.1, b=0.1)

        # Reset the weights of the fc layer based on the mask
        fc_layer.weight.data[weight_mask == 1] = weights_reset[weight_mask == 1]

    def _lrp_fc_layer(self, dataloader):
        fc_layer = self.model.get_last_linear_layer()
        for idx, (inputs, _) in enumerate(dataloader):
            inputs = inputs.to(DEVICE)
            outputs = self.model(inputs)
            T = torch.eye(outputs.size(-1)).to(DEVICE)
            # Select the row from the identity matrix that corresponds to the outputs's highest logit
            T = T[outputs.argmax(dim=1)]
            C = torch.mm(T, fc_layer.weight)
        relevance_per_neuron = C.sum(dim=0)
        normalized_relevance = (relevance_per_neuron - relevance_per_neuron.min()) / (
            relevance_per_neuron.max() - relevance_per_neuron.min()
        )
        normalized_relevance = (normalized_relevance * 2) - 1
        return normalized_relevance

    def _get_mask_relevant_weights(self, relevance_per_neuron, threshold):
        mask_neuron = torch.where(
            relevance_per_neuron >= threshold, torch.tensor(1), torch.tensor(0)
        )
        count_ones = torch.sum(mask_neuron)
        mask_weight = torch.stack([mask_neuron] * self.num_classes, dim=0)
        return mask_weight, count_ones

    def _forget_iterations(self):
        fc_layer = self.model.get_last_linear_layer()
        # Freeze all layers except the fc_layer
        for param in self.model.parameters():
            param.requires_grad = False
        for param in fc_layer.parameters():
            param.requires_grad = True

        zap_optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.model.train()
        for it in range(self.zap_iterations):
            for inputs, _ in self.dl["forget"]:
                inputs = inputs.to(DEVICE, non_blocking=True)
                zap_optimizer.zero_grad()
                outputs = self.model(inputs)

                if it == self.zap_iterations - 1:
                    # Unfreeze all layers
                    for param in self.model.parameters():
                        param.requires_grad = True

                # Apply softmax function to output
                output_probs = torch.softmax(outputs, dim=1)
                target_probs = 1 / output_probs
                target_probs = target_probs / target_probs.sum(dim=1, keepdim=True)

                zap_loss = torch.mean(torch.sum(-target_probs * output_probs, dim=1))
                zap_loss.backward()
                zap_optimizer.step()
