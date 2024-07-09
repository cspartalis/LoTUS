"""
This file contains the implementation of Naive Unlearning Methods:
- Finetuning
- NegGrad
- Relabel
"""

import random
import time

import mlflow
import torch
from tqdm import tqdm

from eval import compute_accuracy, log_membership_attack_prob
from seed import set_work_init_fn
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
            parent_instance.dataset,
            parent_instance.seed,
        )
        self.is_multi_label = True if parent_instance.dataset == "mucac" else False
        if self.is_multi_label:
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
        else:
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

        mlflow.log_param("loss", "cross_entropy")
        mlflow.log_param("lr", self.lr)
        mlflow.log_param("momentum", self.momentum)
        mlflow.log_param("weight_decay", self.weight_decay)
        mlflow.log_param("optimizer", self.optimizer)
        mlflow.log_param("lr_scheduler", "None")

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

            acc_retain = compute_accuracy(
                self.model, self.dl["retain"], self.is_multi_label
            )
            acc_forget = compute_accuracy(
                self.model, self.dl["forget"], self.is_multi_label
            )

            mlflow.log_metric("acc_retain", acc_retain, step=(epoch + 1))
            mlflow.log_metric("acc_forget", acc_forget, step=(epoch + 1))

            log_membership_attack_prob(
                self.dl["retain"],
                self.dl["forget"],
                self.dl["test"],
                self.dl["val"],
                self.model,
                step=(epoch + 1),
            )

        return self.model, run_time

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

            acc_retain = compute_accuracy(self.model, self.dl["retain"], False)
            acc_forget = compute_accuracy(self.model, self.dl["forget"], False)

            # Log accuracies
            mlflow.log_metric("acc_retain", acc_retain)
            mlflow.log_metric("acc_forget", acc_forget)

            log_membership_attack_prob(
                self.dl["retain"],
                self.dl["forget"],
                self.dl["test"],
                self.dl["val"],
                self.model,
                step=(epoch + 1),
            )

        return self.model, run_time


    def _relabel_if_not_multilabel(self):
        run_time = 0  # pylint: disable=invalid-name
        for epoch in tqdm(range(self.epochs)):
            start_time = time.time()
            self.model.train()

            random_forget_labels = []
            forget_inputs = []
            for x, y in self.dl["forget"].dataset:
                rnd = random.randint(0, self.num_classes - 1)
                while rnd == y:
                    rnd = random.randint(0, self.num_classes - 1)
                random_forget_labels.append(rnd)
                forget_inputs.append(x)
            forget_inputs = torch.stack(forget_inputs).cpu()
            random_forget_labels = torch.tensor(random_forget_labels).cpu()
            random_forget_dataset = torch.utils.data.TensorDataset(
                forget_inputs, random_forget_labels
            )

            random_forget_dl = torch.utils.data.DataLoader(
                random_forget_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                worker_init_fn=set_work_init_fn(seed=self.seed),
                num_workers=4,
            )

            # Impair stage
            for inputs, targets in random_forget_dl:
                inputs = inputs.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()

            # Repair stage: Finetune on the "retain" set
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

            acc_retain = compute_accuracy(self.model, self.dl["retain"], False)
            acc_forget = compute_accuracy(self.model, self.dl["forget"], False)

            # Log accuracies
            mlflow.log_metric("acc_retain", acc_retain)
            mlflow.log_metric("acc_forget", acc_forget)

            log_membership_attack_prob(
                self.dl["retain"],
                self.dl["forget"],
                self.dl["test"],
                self.dl["val"],
                self.model,
                step=(epoch + 1),
            )


        return self.model, run_time

    def _relabel_if_multilabel(self, seed):
        run_time = 0  # pylint: disable=invalid-name

        start_dl_prep_time = time.time()
        input_batches = []
        targets_batches = []
        print("Relabeling the forget set")
        for inputs, targets in tqdm(self.dl["forget"]):
            targets = targets.to(DEVICE, non_blocking=True)
            targets = torch.logical_not(targets.bool()).float()
            input_batches.append(inputs)
            targets_batches.append(targets.cpu())
        input_batches = torch.cat(input_batches, dim=0)
        targets_batches = torch.cat(targets_batches, dim=0)
        relabeled_forget_dataset = torch.utils.data.TensorDataset(
            input_batches, targets_batches
        )
        relabeled_forget_dl = torch.utils.data.DataLoader(
            relabeled_forget_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            worker_init_fn=set_work_init_fn(seed=self.seed),
            num_workers=4,
        )
        dl_prep_time = (time.time() - start_dl_prep_time) / 60  # in minutes

        for epoch in tqdm(range(self.epochs)):
            start_time = time.time()
            self.model.train()

            for inputs, targets in relabeled_forget_dl:
                inputs = inputs.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()

            # Repair stage: Finetune on the "retain" set
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

            acc_retain = compute_accuracy(
                self.model, self.dl["retain"], self.is_multi_label
            )
            acc_forget = compute_accuracy(
                self.model, self.dl["forget"], self.is_multi_label
            )

            mlflow.log_metric("acc_retain", acc_retain, step=(epoch + 1))
            mlflow.log_metric("acc_forget", acc_forget, step=(epoch + 1))

        return self.model, run_time + dl_prep_time

    def relabel(self):
        """
        It fine-tunes the model on the "retain" set and on the "relabeled_forget" set

        Returns:
            model (torch.nn.Module): Unlearned model.
            epoch (int): Epoch at which the model was saved.
            run_time (float): Total run time to unlearn the model.
        """
        if self.is_multi_label:
            return self._relabel_if_multilabel()
        return self._relabel_if_not_multilabel()

