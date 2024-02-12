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

from eval import compute_accuracy
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
        )
        self.is_multi_label = True if parent_instance.dataset == "mucac" else False
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
        mlflow.log_param("optimizer", "SGD")
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


            acc_retain = compute_accuracy(self.model, self.dl["retain"], self.is_multi_label)
            acc_forget = compute_accuracy(self.model, self.dl["forget"], self.is_multi_label)
            acc_val = compute_accuracy(self.model, self.dl["val"], self.is_multi_label)

            # Log accuracies
            mlflow.log_metric("acc_retain", acc_retain, step=(epoch + 1))
            mlflow.log_metric("acc_val", acc_val, step=(epoch + 1))
            mlflow.log_metric("acc_forget", acc_forget, step=(epoch + 1))

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

            acc_retain = compute_accuracy(self.model, self.dl["retain"], self.is_multi_label)
            acc_forget = compute_accuracy(self.model, self.dl["forget"], self.is_multi_label)
            acc_val = compute_accuracy(self.model, self.dl["val"], self.is_multi_label)

            # Log accuracies
            mlflow.log_metric("acc_retain", acc_retain, step=(epoch + 1))
            mlflow.log_metric("acc_val", acc_val, step=(epoch + 1))
            mlflow.log_metric("acc_forget", acc_forget, step=(epoch + 1))

        return self.model, run_time

    def neggrad_advanced(self):
        """
        Advanced version of NegGrad.
        Suggested here:
        https://github.com/ndb796/MachineUnlearning/blob/main/01_MUFAC/Machine_Unlearning_MUFAC_NegGrad.ipynb

        Returns:
            model (torch.nn.Module): Unlearned model.
            epoch (int): Epoch at which the model was saved.
            run_time (float): Total run time to unlearn the model.
        """
        run_time = 0  # pylint: disable=invalid-name
        for epoch in tqdm(range(self.epochs)):
            start_time = time.time()
            self.model.train()
            for (inputs_forget, targets_forget), (
                inputs_retrain,
                targets_retain,
            ) in zip(self.dl["forget"], self.dl["retain"]):
                inputs_forget = inputs_forget.to(DEVICE, non_blocking=True)
                targets_forget = targets_forget.to(DEVICE, non_blocking=True)
                inputs_retrain = inputs_retrain.to(DEVICE, non_blocking=True)
                targets_retain = targets_retain.to(DEVICE, non_blocking=True)
                self.optimizer.zero_grad()

                outputs_forget = self.model(inputs_forget)
                outputs_retain = self.model(inputs_retrain)

                loss_forget = -self.loss_fn(outputs_forget, targets_forget)
                loss_retain = self.loss_fn(outputs_retain, targets_retain)

                joint_loss = loss_forget + loss_retain
                joint_loss.backward()
                self.optimizer.step()

            epoch_run_time = (time.time() - start_time) / 60
            run_time += epoch_run_time

            acc_retain = compute_accuracy(self.model, self.dl["retain"], self.is_multi_label)
            acc_forget = compute_accuracy(self.model, self.dl["forget"], self.is_multi_label)
            acc_val = compute_accuracy(self.model, self.dl["val"], self.is_multi_label)

            # Log accuracies
            mlflow.log_metric("acc_retain", acc_retain, step=(epoch + 1))
            mlflow.log_metric("acc_val", acc_val, step=(epoch + 1))
            mlflow.log_metric("acc_forget", acc_forget, step=(epoch + 1))

        return self.model, run_time

    def relabel(self, seed):
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

            ## The commented out snippet doesn't care for not assigning the same label again
            # for inputs, _ in self.dl["forget"]:
            #     inputs = inputs.to(DEVICE, non_blocking=True)

            #     rand_targets = torch.randint(0, self.num_classes, (inputs.size(0),))
            #     rand_targets = rand_targets.squeeze(0).to(DEVICE, non_blocking=True)

            #     self.optimizer.zero_grad()
            #     outputs = self.model(inputs)
            #     loss = self.loss_fn(outputs, rand_targets)
            #     loss.backward()
            #     self.optimizer.step()

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
                worker_init_fn=set_work_init_fn(seed=seed),
                num_workers=4,
            )

            for inputs, targets in random_forget_dl:
                inputs = inputs.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()

            # Impair stage: Finetune on the "retain" set
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

            acc_retain = compute_accuracy(self.model, self.dl["retain"], self.is_multi_label)
            acc_forget = compute_accuracy(self.model, self.dl["forget"], self.is_multi_label)
            acc_val = compute_accuracy(self.model, self.dl["val"], self.is_multi_label)

            # Log accuracies
            mlflow.log_metric("acc_retain", acc_retain, step=(epoch + 1))
            mlflow.log_metric("acc_val", acc_val, step=(epoch + 1))
            mlflow.log_metric("acc_forget", acc_forget, step=(epoch + 1))

        return self.model, run_time

    def relabel_advanced(self, dl_prep_time):
        """
        It fine-tunes the model on the "retain" set and on the "mock_forget" set
        The mock_forget set has samples relabeled to the closest wrong class.

        Returns:
            model (torch.nn.Module): Unlearned model.
            epoch (int): Epoch at which the model was saved.
            run_time (float): Total run time to unlearn the model.
        """
        run_time = 0  # pylint: disable=invalid-name
        for epoch in tqdm(range(self.epochs)):
            start_time = time.time()
            self.model.train()
            for inputs, targets in self.dl["mock_forget"]:
                inputs = inputs.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()

            # Impair stage: Finetune on the "retain" set
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

            acc_retain = compute_accuracy(self.model, self.dl["retain"], self.is_multi_label)
            acc_forget = compute_accuracy(self.model, self.dl["forget"], self.is_multi_label)
            acc_val = compute_accuracy(self.model, self.dl["val"], self.is_multi_label)

            # Log accuracies
            mlflow.log_metric("acc_retain", acc_retain, step=(epoch + 1))
            mlflow.log_metric("acc_val", acc_val, step=(epoch + 1))
            mlflow.log_metric("acc_forget", acc_forget, step=(epoch + 1))

        return self.model, (run_time + dl_prep_time)
