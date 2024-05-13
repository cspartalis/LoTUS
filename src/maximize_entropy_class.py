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
from torch.utils.data import DataLoader, Dataset
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


class UnLearningData(Dataset):
    def __init__(self, forget_data, retain_data):
        super().__init__()
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.forget_len = len(forget_data)
        self.retain_len = len(retain_data)

    def __len__(self):
        return self.retain_len + self.forget_len

    def __getitem__(self, index):
        if index < self.forget_len:
            x = self.forget_data[index][0]
            y = self.forget_data[index][1]
            l = 1
            return x, y, l
        else:
            x = self.retain_data[index - self.forget_len][0]
            y = self.retain_data[index - self.forget_len][1]
            l = 0
            return x, y, l


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
        self.lr = 1e-2
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        self.teacher = copy.deepcopy(parent_instance.model).to(DEVICE)
        mlflow.log_param("lr", self.lr)
        mlflow.log_param("momentum", self.momentum)
        mlflow.log_param("weight_decay", self.weight_decay)
        mlflow.log_param("optimizer", self.optimizer)

    def unlearn(self, is_zapping, is_once, subset_size):

        run_time = 0

        start_prep_time = time.time()

        if is_zapping and is_once:
            self._random_init_weights()

        retrain_subset = self._get_retain_subset(size=subset_size)
        log.info("Retain subset size: %d", len(retrain_subset))
        unlearning_data = UnLearningData(
            forget_data=self.dl["forget"].dataset, retain_data=retrain_subset
        )

        log.info("Creating unlearning dataloader")
        unlearning_dl = DataLoader(
            unlearning_data,
            batch_size=self.batch_size,
            shuffle=True,
            worker_init_fn=set_work_init_fn(self.seed),
            num_workers=4,
        )

        self.teacher.eval()

        prep_time = (time.time() - start_prep_time) / 60  # in minutes

        # Forget trainining
        for epoch in tqdm(range(self.epochs)):
            start_epoch_time = time.time()
            self.model.train()

            if is_zapping and not is_once:
                self._random_init_weights()

            for x, y, l in unlearning_dl:
                x = x.to(DEVICE)
                with torch.no_grad():
                    teacher_logits = self.teacher(x)
                output = self.model(x)
                self.optimizer.zero_grad()
                loss = self.UnlearnerLoss(output, y, l, teacher_logits)
                loss.backward()
                self.optimizer.step()

                # acc_retain = compute_accuracy(self.model, self.dl["retain"], False)
                # acc_forget = compute_accuracy(self.model, self.dl["forget"], False)
                # log.info("Retain %.2f, Forget %.2f", acc_retain, acc_forget)

            epoch_run_time = (time.time() - start_epoch_time) / 60  # in minutes
            run_time += epoch_run_time

            log.info("Computing accuracies")
            acc_retain = compute_accuracy(self.model, self.dl["retain"], False)
            acc_forget = compute_accuracy(self.model, self.dl["forget"], False)

            # Log accuracies
            mlflow.log_metric("acc_retain", acc_retain, step=epoch + 1)
            mlflow.log_metric("acc_forget", acc_forget, step=epoch + 1)

            log.info("Start logging MIA probability")
            log_membership_attack_prob(
                self.dl["retain"],
                self.dl["forget"],
                self.dl["test"],
                self.dl["val"],
                self.model,
                step=(epoch + 1),
            )
            log.info("Experiment completed")

        return self.model, run_time + prep_time

    def _random_init_weights(self) -> None:
        """This function resets the weights of the last fully connected layer of a neural network."""
        fc_layer = self.model.get_last_linear_layer()
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

    def UnlearnerLoss(self, outputs, targets, labels, full_teacher_logits):
        labels = torch.unsqueeze(labels, dim=1).to(DEVICE)

        teacher_out = F.softmax(full_teacher_logits, dim=1)
        # uniform_out = torch.ones_like(teacher_out) / (self.num_classes)
        uniform_out = torch.ones_like(teacher_out) / (self.num_classes - 1)

        indices = np.arange(uniform_out.shape[0])
        uniform_out[indices, targets] = 0

        # label 1 means forget sample
        # label 0 means retain sample
        overall_out = labels * uniform_out + (1 - labels) * teacher_out
        student_out = F.log_softmax(outputs, dim=1)

        cross_entropy_loss = -torch.sum(overall_out * student_out, dim=1)
        mean_loss = torch.mean(cross_entropy_loss)

        return mean_loss
