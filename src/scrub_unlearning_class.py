"""
This file contains the implementation of SCRUB:
https://arxiv.org/pdf/2302.09880.pdf
"""

import copy
import time

import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from unlearning_base_class import UnlearningBaseClass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DistillKL(nn.Module):
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss


class SCRUB(UnlearningBaseClass):
    """
    Code from the repo: https://github.com/meghdadk/SCRUB
    The hyperparameters are the same as the suggested in the original codebase.
    """

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
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss()
        self.lr = 5e-4
        if self.dataset == "cifar-100":
            self.weight_decay = 5e-4 # weight decay for large-scale datasets 
        else:
            self.weight_decay = 0.1
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.criterion_div = DistillKL(4.0)
        self.criterion_kd = DistillKL(4.0)

        mlflow.log_param("loss", "cross_entropy")
        mlflow.log_param("lr", self.lr)
        # mlflow.log_param("momentum", self.momentum)
        mlflow.log_param("weight_decay", self.weight_decay)
        mlflow.log_param("optimizer", self.optimizer)
        mlflow.log_param("lr_scheduler", "None")

    def unlearn(self):
        run_time = 0  # pylint: disable=invalid-name
        teacher = copy.deepcopy(self.model)
        student = copy.deepcopy(self.model)

        for epoch in tqdm(range(self.epochs)):
            start_time = time.time()
            teacher.train()
            student.train()

            # Training with retain data
            for inputs_retain, labels_retain in self.dl["retain"]:
                inputs_retain = inputs_retain.to(DEVICE, non_blocking=True)
                labels_retain = labels_retain.to(DEVICE, non_blocking=True)

                # Forward pass: Student
                outputs_retain_student = student(inputs_retain)

                # Forward pass: Teacher
                with torch.no_grad():
                    outputs_retain_teacher = teacher(inputs_retain)

                # Loss computation
                loss_cls = self.loss_fn(outputs_retain_student, labels_retain)
                loss_div_retain = self.criterion_div(
                    outputs_retain_student, outputs_retain_teacher
                )

                loss = loss_cls + loss_div_retain

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Training with forget data
            for inputs_forget, labels_forget in self.dl["forget"]:
                inputs_forget = inputs_forget.to(DEVICE, non_blocking=True)
                labels_forget = labels_forget.to(DEVICE, non_blocking=True)

                # Forward pass: Student
                outputs_forget_student = student(inputs_forget)

                # Forward pass: Teacher
                with torch.no_grad():
                    outputs_forget_teacher = teacher(inputs_forget)

                # We want to maximize the divergence for the forget data.
                loss_div_forget = -self.criterion_div(
                    outputs_forget_student, outputs_forget_teacher
                )

                # Backward pass
                self.optimizer.zero_grad()
                loss_div_forget.backward()
                self.optimizer.step()

            epoch_run_time = (time.time() - start_time) / 60  # in minutes
            run_time += epoch_run_time

        return self.model, run_time
