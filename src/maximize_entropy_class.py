"""
This file contains the implementation of Zap Unlearning
Our proposed method
"""

import copy
import logging
import os
import time

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

from eval import (
    compute_accuracy,
    get_membership_attack_data,
    log_membership_attack_prob,
)
from seed import set_work_init_fn
from unlearning_base_class import UnlearningBaseClass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log = logging.getLogger(__name__)


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


class MaximizeEntropy(UnlearningBaseClass):
    def __init__(self, parent_instance):
        super().__init__(
            parent_instance.dl,
            parent_instance.batch_size,
            parent_instance.num_classes,
            parent_instance.model,
            parent_instance.epochs,
            parent_instance.dataset,
            parent_instance.seed,
        )
        self.is_multi_label = True if parent_instance.dataset == "mucac" else False
        if self.is_multi_label:
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()
        self.soft_loss_fn = SoftCrossEntropyLoss(self.num_classes, self.dataset)
        self.lr = 1e-3
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        self.impair_optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=1e-3,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        self.lr_scheduler = None
        mlflow.log_param("loss", self.loss_fn)
        mlflow.log_param("lr", self.lr)
        mlflow.log_param("momentum", self.momentum)
        mlflow.log_param("weight_decay", self.weight_decay)
        mlflow.log_param("optimizer", self.optimizer)
        mlflow.log_param("impair_optimizer", self.impair_optimizer)
        mlflow.log_param("lr_scheduler", self.lr_scheduler)

    def unlearn(self, is_zapping, is_once, str_forget_loss, subset_size):

        if str_forget_loss == "ce":
            forget_loss = self.soft_loss_fn
        elif str_forget_loss == "kl":
            forget_loss = self.kl_loss_fn
        else:
            raise ValueError("Forget loss must be either 'ce' or 'kl'")

        run_time = 0

        start_prep_time = time.time()

        if is_zapping and is_once:
            self._random_init_weights()

        retrain_subset = self._get_retain_subset(size=subset_size)
        retain_subset_dl = torch.utils.data.DataLoader(
            retrain_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            worker_init_fn=set_work_init_fn(self.seed),
        )

        prep_time = (time.time() - start_prep_time) / 60  # in minutes

        # Forget trainining
        for epoch in tqdm(range(self.epochs)):
            start_epoch_time = time.time()
            self.model.train()

            if is_zapping and not is_once:
                self._random_init_weights()

            log.info("Start the impair step")
            # Impair phase
            for x, _ in self.dl["forget"].dataset:
                x = x.unsqueeze(0).to(DEVICE)
                self.impair_optimizer.zero_grad()
                self.model.zero_grad()
                output = self.model(x)
                loss = forget_loss(output)
                loss.backward()
                self.impair_optimizer.step()
            log.info("End the impair step with run time: %s", run_time)

            epoch_run_time = (time.time() - start_epoch_time) / 60  # in minutes

            acc_retain = compute_accuracy(
                self.model, self.dl["retain"], self.is_multi_label
            )
            acc_forget = compute_accuracy(
                self.model, self.dl["forget"], self.is_multi_label
            )

            # Log accuracies
            mlflow.log_metric("acc_retain", acc_retain, step=(epoch + 1))
            mlflow.log_metric("acc_forget", acc_forget, step=(epoch + 1))

            # log_membership_attack_prob(
            #     self.dl["retain"],
            #     self.dl["forget"],
            #     self.dl["test"],
            #     self.dl["val"],
            #     self.model,
            #     step=(epoch + 1),
            # )

            # Repair phase
            for inputs, targets in retain_subset_dl:
                inputs = inputs.to(DEVICE, non_blocking=True)
                targets = torch.tensor(targets).to(DEVICE, non_blocking=True)
                self.optimizer.zero_grad()
                self.model.zero_grad()
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

            # Log accuracies
            mlflow.log_metric("acc_retain", acc_retain, step=(epoch + 2))
            mlflow.log_metric("acc_forget", acc_forget, step=(epoch + 2))

        log_membership_attack_prob(
            self.dl["retain"],
            self.dl["forget"],
            self.dl["test"],
            self.dl["val"],
            self.model,
            step=(epoch + 2),
        )

        return self.model, run_time + prep_time

    def _random_init_weights(self) -> None:
        """This function resets the weights of the last fully connected layer of a neural network."""
        fc_layer = self.model.get_last_linear_layer()
        if self.is_multi_label:
            fc_layer = fc_layer.fc
        nn.init.kaiming_normal_(tensor=fc_layer.weight.data, nonlinearity="relu")
        fc_layer.bias.data.zero_()

    def _get_retain_subset(self, size):
        # creating the unlearning dataset.
        indices = list(range(len(self.dl["retain"].dataset)))
        targets = [y for _, y in self.dl["retain"].dataset]

        stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=size)
        _, sample_indices = next(stratified_split.split(indices, targets))

        retain_train_subset = torch.utils.data.Subset(
            self.dl["retain"].dataset, sample_indices
        )
        return retain_train_subset
