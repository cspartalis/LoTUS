"""
This file contains the implementation of Naive Unlearning Methods:
- Finetuning
- NegGrad
- Relabel
"""

import time

import mlflow
import torch
from tqdm import tqdm

from eval import compute_accuracy
from unlearning_base_class import UnlearningBaseClass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NaiveUnlearning(UnlearningBaseClass):
    def __init__(self, parent_instance):
        super().__init__(
            parent_instance.dl,
            parent_instance.batch_size,
            parent_instance.num_classes,
            parent_instance.model,
            parent_instance.epochs,
            parent_instance.acc_forget_retrain,
            parent_instance.is_early_stop,
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()
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

    def finetune(self):
        """
        Finetune the model on the "retain" set.

        Returns:
            model (torch.nn.Module): Unlearned model.
            epoch (int): Epoch at which the model was saved.
            run_time (float): Total run time to unlearn the model.
        """
        run_time = 0  # pylint: disable=invalid-name
        for epoch in tqdm(range(self.epochs)):
            start_time = time.time()
            self.model.train()
            for inputs, targets in self.dl["retain"]:
                inputs = inputs.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()
            epoch_run_time = (time.time() - start_time) / 60  # in minutes
            run_time += epoch_run_time

            acc_retain = compute_accuracy(self.model, self.dl["retain"])
            acc_forget = compute_accuracy(self.model, self.dl["forget"])
            acc_val = compute_accuracy(self.model, self.dl["val"])

            # Log accuracies
            mlflow.log_metric("acc_retain", acc_retain, step=(epoch + 1))
            mlflow.log_metric("acc_val", acc_val, step=(epoch + 1))
            mlflow.log_metric("acc_forget", acc_forget, step=(epoch + 1))

            if self.is_early_stop:
                if acc_forget <= self.acc_forget_retrain:
                    return self.model, epoch, run_time

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        return self.model, self.epochs, run_time

    def neggrad(self):
        """
        Finetune the model on the "forget" set using gradient ascent
        and on the "retain" set using gradient descent.

        Returns:
            model (torch.nn.Module): Unlearned model.
            epoch (int): Epoch at which the model was saved.
            run_time (float): Total run time to unlearn the model.
        """
        run_time = 0  # pylint: disable=invalid-name
        for epoch in tqdm(range(self.epochs)):
            start_time = time.time()
            self.model.train()
            for inputs, targets in self.dl["forget"]:
                inputs = inputs.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad *= -1
                loss.backward()
                self.optimizer.step()

            for inputs, targets in self.dl["retain"]:
                inputs = inputs.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()

            epoch_run_time = (time.time() - start_time) / 60
            run_time += epoch_run_time

            acc_retain = compute_accuracy(self.model, self.dl["retain"])
            acc_forget = compute_accuracy(self.model, self.dl["forget"])
            acc_val = compute_accuracy(self.model, self.dl["val"])

            # Log accuracies
            mlflow.log_metric("acc_retain", acc_retain, step=(epoch + 1))
            mlflow.log_metric("acc_val", acc_val, step=(epoch + 1))
            mlflow.log_metric("acc_forget", acc_forget, step=(epoch + 1))

            if self.is_early_stop:
                if acc_forget <= self.acc_forget_retrain:
                    return self.model, epoch, run_time

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        return self.model, self.epochs, run_time

    def relabel_advanced(self, dl_prep_time):
        """
        It fine-tunes the model on the "retain" set and on the "relabeled_forget" set

        Returns:
            model (torch.nn.Module): Unlearned model.
            epoch (int): Epoch at which the model was saved.
            run_time (float): Total run time to unlearn the model.
        """
        run_time = 0  # pylint: disable=invalid-name
        for epoch in tqdm(range(self.epochs)):
            start_time = time.time()
            self.model.train()
            for inputs, targets in self.dl["mixed"]:
                inputs = inputs.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()
            epoch_run_time = (time.time() - start_time) / 60  # in minutes
            run_time += epoch_run_time

            acc_retain = compute_accuracy(self.model, self.dl["retain"])
            acc_forget = compute_accuracy(self.model, self.dl["forget"])
            acc_val = compute_accuracy(self.model, self.dl["val"])

            # Log accuracies
            mlflow.log_metric("acc_retain", acc_retain, step=epoch)
            mlflow.log_metric("acc_val", acc_val, step=epoch)
            mlflow.log_metric("acc_forget", acc_forget, step=epoch)

            if self.is_early_stop:
                if acc_forget <= self.acc_forget_retrain:
                    return self.model, epoch, run_time + dl_prep_time

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        return self.model, self.epochs, run_time + dl_prep_time

    def relabel(self):
        """
        It fine-tunes the model on the "retain" set and on the "relabeled_forget" set

        Returns:
            model (torch.nn.Module): Unlearned model.
            epoch (int): Epoch at which the model was saved.
            run_time (float): Total run time to unlearn the model.
        """
        run_time = 0  # pylint: disable=invalid-name
        for epoch in tqdm(range(self.epochs)):
            start_time = time.time()
            self.model.train()
            for inputs, _ in self.dl["forget"]:
                inputs = inputs.to(DEVICE, non_blocking=True)

                rand_targets = torch.randint(0, self.num_classes, (inputs.size(0),))
                rand_targets = rand_targets.squeeze(0).to(DEVICE, non_blocking=True)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, rand_targets)
                loss.backward()
                self.optimizer.step()
            epoch_run_time = (time.time() - start_time) / 60  # in minutes
            run_time += epoch_run_time

            # Impair stage: Finetune on the "retain" set
            for inputs, targets in self.dl["retain"]:
                inputs = inputs.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                ### Debugging
                # print("outputs: ", outputs[0])
                # print("targets[0]: ", targets[0])
                # softmax_outputs = torch.nn.functional.softmax(outputs[0])
                # print("softmax_outputs: ", softmax_outputs)
                # print("torch.sum(softmax_outputs): ", torch.sum(softmax_outputs))
                # log_softmax_outputs = torch.nn.functional.log_softmax(outputs[0])
                # print("log_softmax_outputs: ", log_softmax_outputs)
                # print("torch.sum(log_softmax_outputs): ", torch.sum(log_softmax_outputs))
                # loss = self.loss_fn(outputs[0], targets[0])
                # print("loss: ", loss)
                # exit()
                ###
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()
