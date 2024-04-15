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


class MaximizeEntropy(UnlearningBaseClass):
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

    def unlearn(self, is_zapping, is_once, str_forget_loss):

        if str_forget_loss == "ce":
            forget_loss = self.soft_loss_fn
        elif str_forget_loss == "kl":
            forget_loss = self.kl_loss_fn
        else:
            raise ValueError("Forget loss must be either 'ce' or 'kl'")

        run_time = 0

        if is_zapping and is_once:
            self._random_init_weights()

        # Forget trainining
        for epoch in tqdm(range(self.epochs)):
            start_epoch_time = time.time()
            self.model.train()

            if is_zapping and not is_once:
                self._random_init_weights()

            # Impair phase
            for inputs, _ in self.dl["forget"]:
                inputs = inputs.to(DEVICE, non_blocking=True)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = forget_loss(outputs)
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

        return self.model, run_time

    def _random_init_weights(self) -> None:
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
