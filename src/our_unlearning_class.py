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
    def __init__(self, num_classes, dataset):
        super().__init__()
        self.num_classes = num_classes
        self.dataset = dataset

    def forward(self, outputs):
        if self.dataset != "mucac":
            log_probs = torch.log_softmax(outputs, dim=1)
            uniform_probs = torch.ones_like(log_probs) / self.num_classes
            uniform_probs = uniform_probs.to(DEVICE, non_blocking=True)
        else:
            log_sigmoid = torch.nn.LogSigmoid()
            log_probs = log_sigmoid(outputs)
            uniform_probs = torch.ones_like(log_probs) * 0.5
            uniform_probs = uniform_probs.to(DEVICE, non_blocking=True)
        loss = -torch.sum(uniform_probs * log_probs, dim=1)
        return torch.mean(loss)


class KLLoss(nn.Module):
    """
    https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html#KLDivLoss
    """

    def __init__(self, num_classes, dataset):
        super().__init__()
        self.num_classes = num_classes
        self.dataset = dataset

    def forward(self, outputs):
        if self.dataset != "mucac":
            probs = torch.softmax(outputs, dim=1)
            uniform_probs = torch.ones_like(probs) / self.num_classes
            uniform_probs = uniform_probs.to(DEVICE, non_blocking=True)
        else:
            probs = torch.sigmoid(outputs)
            uniform_probs = torch.ones_like(probs) * 0.5
            uniform_probs = uniform_probs.to(DEVICE, non_blocking=True)
        # log_target=True is important, otherwise F.kl_div(P||P) != 0
        kl_divergence = F.kl_div(
            probs, uniform_probs, reduction="batchmean", log_target=True
        )
        return kl_divergence


class OurUnlearning(UnlearningBaseClass):
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
        if self.is_multi_label:
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()
        self.soft_loss_fn = SoftCrossEntropyLoss(self.num_classes, self.dataset)
        self.kl_loss_fn = KLLoss(self.num_classes, self.dataset)
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
        mlflow.log_param("loss", self.loss_fn)
        mlflow.log_param("lr", self.lr)
        mlflow.log_param("momentum", self.momentum)
        mlflow.log_param("weight_decay", self.weight_decay)
        mlflow.log_param("optimizer", self.optimizer)
        mlflow.log_param("lr_scheduler", self.lr_scheduler)

    ####################################################################################################
    ############################## L R P - SOFT CROSS ENTROPY LOSS #####################################
    ####################################################################################################
    def our_lrp_ce(self, relevance_threshold):
        mlflow.log_param("relevance_threshold", relevance_threshold)
        run_time = 0

        # Zap the weights of the weights of the defined neurons.
        start_prep_time = time.time()
        neuron_contrib_forget = self._lrp_importances(self.dl["forget"])
        neuron_contrib_retain = self._lrp_importances(self.dl["retain"])
        neuron_contrib_diff = neuron_contrib_forget - neuron_contrib_retain
        scaled_neuron_contrib = self._custom_scaling(neuron_contrib_diff)

        mask_weight, rel_weights = self._get_mask_for_weights_lrp(
            scaled_neuron_contrib, threshold=relevance_threshold
        )
        prep_time = (time.time() - start_prep_time) / 60  # in minutes
        mlflow.log_param("rel_weights", rel_weights.item())

        # Forget trainining
        for epoch in tqdm(range(self.epochs)):
            start_epoch_time = time.time()

            self._random_init_weights(mask_weight)
            self.model.train()

            # Impair phase
            for inputs, _ in self.dl["forget"]:
                inputs = inputs.to(DEVICE, non_blocking=True)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.soft_loss_fn(outputs)
                loss.backward()
                self.optimizer.step()

            # Repair phase
            for inputs, targets in self.dl["retain"]:
                inputs = inputs.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
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

        return self.model, run_time + prep_time

    ####################################################################################################
    ############################## F I M - SOFT CROSS ENTROPY LOSS #####################################
    ####################################################################################################

    def our_fim_ce(self, relevance_threshold):
        mlflow.log_param("relevance_threshold", relevance_threshold)
        run_time = 0

        # Zap the weights of the weights of the defined neurons.
        start_prep_time = time.time()
        weight_contrib_forget = self._fim_importances(self.dl["forget"])
        weight_contrib_retain = self._fim_importances(self.dl["retain"])
        weight_contrib_diff = weight_contrib_forget - weight_contrib_retain
        scaled_weight_contrib = self._custom_scaling(weight_contrib_diff)

        mask_weight, rel_weights = self._get_mask_for_weights_fim(
            scaled_weight_contrib, threshold=relevance_threshold
        )
        prep_time = (time.time() - start_prep_time) / 60  # in minutes
        mlflow.log_param("rel_weights", rel_weights.item())

        # Forget trainining
        for epoch in tqdm(range(self.epochs)):
            start_epoch_time = time.time()

            self._random_init_weights(mask_weight)
            self.model.train()

            # Impair phase
            for inputs, _ in self.dl["forget"]:
                inputs = inputs.to(DEVICE, non_blocking=True)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.soft_loss_fn(outputs)
                loss.backward()
                self.optimizer.step()

            # Repair phase
            for inputs, targets in self.dl["retain"]:
                inputs = inputs.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
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

        return self.model, run_time + prep_time

    ####################################################################################################
    ############################## L R P - KULLBACK-LEIBLER DIVERGENCE LOSS ############################
    ####################################################################################################
    def our_lrp_kl(self, relevance_threshold):
        mlflow.log_param("relevance_threshold", relevance_threshold)
        run_time = 0

        # Zap the weights of the weights of the defined neurons.
        start_prep_time = time.time()
        neuron_contrib_forget = self._lrp_importances(self.dl["forget"])
        neuron_contrib_retain = self._lrp_importances(self.dl["retain"])
        neuron_contrib_diff = neuron_contrib_forget - neuron_contrib_retain
        scaled_neuron_contrib = self._custom_scaling(neuron_contrib_diff)

        mask_weight, rel_weights = self._get_mask_for_weights_lrp(
            scaled_neuron_contrib, threshold=relevance_threshold
        )
        prep_time = (time.time() - start_prep_time) / 60  # in minutes
        mlflow.log_param("rel_weights", rel_weights.item())

        # Forget trainining
        for epoch in tqdm(range(self.epochs)):
            start_epoch_time = time.time()

            self._random_init_weights(mask_weight)
            self.model.train()

            # Impair phase
            for inputs, _ in self.dl["forget"]:
                inputs = inputs.to(DEVICE, non_blocking=True)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.kl_loss_fn(outputs)
                loss.backward()
                self.optimizer.step()

            # Repair phase
            for inputs, targets in self.dl["retain"]:
                inputs = inputs.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
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

        return self.model, run_time + prep_time

    ####################################################################################################
    ############################## F I M - KULLBACK-LEIBLER DIVERGENCE LOSS ############################
    ####################################################################################################
    def our_fim_kl(self, relevance_threshold):
        mlflow.log_param("relevance_threshold", relevance_threshold)
        run_time = 0

        start_prep_time = time.time()
        # Zap the weights of the weights of the defined neurons.
        weight_contrib_forget = self._fim_importances(self.dl["forget"])
        weight_contrib_retain = self._fim_importances(self.dl["retain"])
        weight_contrib_diff = weight_contrib_forget - weight_contrib_retain
        scaled_weight_contrib = self._custom_scaling(weight_contrib_diff)

        mask_weight, rel_weights = self._get_mask_for_weights_fim(
            scaled_weight_contrib, threshold=relevance_threshold
        )
        prep_time = (time.time() - start_prep_time) / 60  # in minutes
        mlflow.log_param("rel_weights", rel_weights.item())

        # Forget trainining
        for epoch in tqdm(range(self.epochs)):
            start_epoch_time = time.time()

            self._random_init_weights(mask_weight)
            self.model.train()

            # Impair phase
            for inputs, _ in self.dl["forget"]:
                inputs = inputs.to(DEVICE, non_blocking=True)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.kl_loss_fn(outputs)
                loss.backward()
                self.optimizer.step()

            # Repair phase
            for inputs, targets in self.dl["retain"]:
                inputs = inputs.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
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

        return self.model, run_time + prep_time

    ####################################################################################################
    ########################################### H E L P E R S ##########################################
    ####################################################################################################

    def _lrp_importances(self, dataloader):
        fc_layer = self.model.get_last_linear_layer()
        if self.is_multi_label:
            fc_layer = fc_layer.fc
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

    def _get_mask_for_weights_lrp(self, relevance_per_neuron, threshold):
        mask_neuron = torch.where(
            relevance_per_neuron >= threshold, torch.tensor(1), torch.tensor(0)
        )
        mask_weight = torch.stack([mask_neuron] * self.num_classes, dim=0)
        count_ones = torch.sum(mask_weight)
        return mask_weight, count_ones

    def _random_init_weights(self, weight_mask) -> None:
        """
        This function resets the weights of the last fully connected layer of a neural network.
        Args:
            weight_mask (torch.Tensor): It contains ones for the weights to be zapped, zeros for the others.
        """
        fc_layer = self.model.get_last_linear_layer()
        if self.is_multi_label:
            fc_layer = fc_layer.fc
        # Get the weights of the fc layer
        weights_reset = fc_layer.weight.data.detach().clone()
        torch.nn.init.xavier_normal_(tensor=weights_reset, gain=1.0)
        # Reset the weights of the fc layer based on the mask
        fc_layer.weight.data[weight_mask == 1] = weights_reset[weight_mask == 1]

    def _fim_importances(self, dataloader):
        model = copy.deepcopy(self.model)
        fc_layer = model.get_last_linear_layer()
        if self.is_multi_label:
            fc_layer = fc_layer.fc
        criterion = self.loss_fn
        importances = dict(
            [
                (k, torch.zeros_like(p, device=p.device))
                for k, p in fc_layer.named_parameters()
            ]
        )
        for batch in dataloader:
            x, y = batch
            x, y = x.to(DEVICE), y.to(DEVICE)
            self.optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()

            for (k1, p), (k2, imp) in zip(
                fc_layer.named_parameters(), importances.items()
            ):
                if p.grad is not None:
                    imp.data += p.grad.data.clone().pow(2)

        # average over mini batch length
        for _, imp in importances.items():
            imp.data /= float(len(dataloader))
        return importances["weight"]

    def _get_mask_for_weights_fim(self, weight_contrib, threshold):
        mask_weight = torch.where(
            weight_contrib >= threshold, torch.tensor(1), torch.tensor(0)
        )
        count_ones = torch.sum(mask_weight)
        return mask_weight, count_ones

    def _custom_scaling(self, x):
        """
        Custom minmax scaling, where the positives are scaled to [0, 1]
        and the negatives are scaled to [-1, 0], and zeros remain zeros.
        """
        scaled_x = x.clone()
        pos_mask = x > 0
        neg_mask = x < 0
        zero_mask = ~(pos_mask | neg_mask)

        # print(pos_mask.sum(), neg_mask.sum(), zero_mask.sum())

        # Calculate scaling factors for positive and negative values
        if pos_mask.sum() > 0:
            pos_scale = 1 / torch.max(x[pos_mask])
            scaled_x[pos_mask] = x[pos_mask] * pos_scale
        if neg_mask.sum() > 0:
            neg_scale = 1 / torch.abs(torch.min(x[neg_mask]))
            scaled_x[neg_mask] = x[neg_mask] * neg_scale

        scaled_x[zero_mask] = 0.0  # Ensure zeros remain zeros

        return scaled_x
