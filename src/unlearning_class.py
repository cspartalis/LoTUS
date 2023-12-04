"""
This file contains the implementation of the unlearning methods.
"""

# pylint: disable=import-error
import copy
import time

import mlflow
import torch
from tqdm import tqdm

import boundary_utils as bu
import zapping_utils as zu
from eval import compute_accuracy

# pylint: enable=import-error

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class UnlearningClass:
    """
    Class representing an unlearning method.

    Args:
        dl (dict): Data loader object containing different datasets.
        batch_size (int): Batch size.
        num_classes (int): Number of classes in the dataset.
        model: Model to train.
        epochs (int): Number of epochs to train.
        loss_fn: Loss function.
        optimizer: Optimizer.
        lr_scheduler: Learning rate scheduler (it could be None).
        acc_forget_retrain: Accuracy threshold for retraining.
        is_early_stop (bool): Whether to use early stopping.

    Methods:
        finetune:  Fine-tunes the model on the "retain" set.
        neggrad:   Fine-tunes the model on the "forget" set using gradient ascent
                    and on the "retain" set using gradient descent.
        relabel:   Fine-tunes the model on the "relabeled_forget" set and on the "retain" set.
        boundary:  Perfrom Pgd attack on the "forget" set and fine-tunes the model
                    as described in the paper "Boundary Unlearning".
        zapping:   Our implementation. Fine-tunes the model on the "retain" set and on the
                    "relabeled_forget" set after zapping (some or all) of the fc neurons.

        get_relabeled_forget: Relabels the "forget" set based on the highest logit that doesn't
                               correspond to the true label.

    Returns:
        model (torch.nn.Module): Unlearned model.
        epoch (int): Epoch at which the model was unlearned.
        run_time (float): Total run time to unlearn the model.
    """

    def __init__(
        self,
        dl,
        batch_size,
        num_classes,
        model,
        epochs,
        loss_fn,
        optimizer,
        lr_scheduler,
        acc_forget_retrain,
        is_early_stop,
    ) -> None:
        self.dl = dl
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.model = model
        self.epochs = epochs
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.acc_forget_retrain = acc_forget_retrain
        self.is_early_stop = is_early_stop

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

    def relabel(self, dl_prep_time):
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

    def boundary(self):
        """
        Code from the paper "Boundary Unlearning".
        All the var values are the same as in the code from the paper.
        Whereverer there are snippets of my code, I have added a corresponding comment.
        The arguments that should be passed to the function to match the paper are:
        - loss = cross_entropy
        - optimizer = sgd
        - lr = 1e-5
        - momentum = 0.9
        - weight_decay = 0
        - is_lr_scheduler = False

        Returns:
            model (torch.nn.Module): Unlearned model.
            epoch (int): Epoch at which the model was saved.
            run_time (float): Total run time to unlearn the model.
        """

        # start of my snippet
        start_prep_time = time.time()
        # end of my snippet

        test_model = copy.deepcopy(self.model).to(DEVICE)
        unlearn_model = copy.deepcopy(self.model).to(DEVICE)

        adv = bu.FGSM(
            test_model, bound=0.1, norm=True, random_start=False, device=DEVICE
        )
        forget_data_gen = bu.inf_generator(self.dl["forget"])
        batches_per_epoch = len(self.dl["forget"])
        prep_time = (time.time() - start_prep_time) / 60

        num_hits = 0
        num_sum = 0
        nearest_label = []
        run_time = 0  # pylint: disable=invalid-name
        for epoch in tqdm(range(self.epochs)):
            start_time = time.time()
            self.model.train()
            for itr in range(batches_per_epoch):
                x, y = forget_data_gen.__next__()
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                test_model.eval()
                x_adv = adv.perturb(
                    x, y, target_y=None, model=test_model, device=DEVICE
                )
                adv_logits = test_model(x_adv)
                pred_label = torch.argmax(adv_logits, dim=1)
                if itr >= batches_per_epoch - 1:
                    nearest_label.append(pred_label.tolist())
                num_hits += (y != pred_label).float().sum()
                num_sum += y.shape[0]

                # adv_train
                unlearn_model.train()
                unlearn_model.zero_grad()
                self.optimizer.zero_grad()

                ori_logits = unlearn_model(x)
                loss = self.loss_fn(ori_logits, pred_label)

                loss.backward()
                self.optimizer.step()

            # start of my snippet
            epoch_run_time = (time.time() - start_time) / 60  # in minutes
            run_time += epoch_run_time

            # start of my snippet
            acc_retain = compute_accuracy(self.model, self.dl["retain"])
            acc_forget = compute_accuracy(self.model, self.dl["forget"])
            acc_val = compute_accuracy(self.model, self.dl["val"])

            # Log accuracies
            mlflow.log_metric("acc_retain", acc_retain, step=(epoch + 1))
            mlflow.log_metric("acc_val", acc_val, step=(epoch + 1))
            mlflow.log_metric("acc_forget", acc_forget, step=(epoch + 1))

            if self.is_early_stop:
                if acc_forget <= self.acc_forget_retrain:
                    return self.model, epoch, run_time + prep_time

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # end of my snippet

        return self.model, self.epochs, run_time + prep_time

    def zapping(self, is_diff_grads, threshold, dl_prep_time):
        """
        It fine-tunes the model on the "retain" set and on the
        "relabeled_forget" set after re-initiallizing some weights of the fc_layer.
        This is based on the gradients fo the weights in the fc_layer.
        The zapped weights can be those that contribute the most for the "forget" set (hard zapping),
        or those that are both the most important for the "forget" set and the least important
        for the "retain" set (soft zapping).

        Args:
            is_diff_grads (bool): _description_
            threshold (float, optional): _description_. Defaults to 0.5.

        Returns:
            model: Unlearned model.
            epoch: Epoch at which the model was saved.
            run_time: Total run time to unlearn the model.
        """
        start_prep_time = time.time()
        forget_grads = zu.get_fc_gradients(self.model, self.dl["forget"], self.loss_fn)
        if is_diff_grads:  # True: soft zapping, False: hard zapping
            retain_grads = zu.get_fc_gradients(
                self.model, self.dl["retain"], self.loss_fn
            )
            diff_grads = zu.get_diff_gradients(forget_grads, retain_grads)
            weight_mask = zu.get_weight_mask(diff_grads, threshold)
            zu.visualize_fc_grads(diff_grads, "diff_grads")
        else:
            weight_mask = zu.get_weight_mask(forget_grads, threshold)
            zu.visualize_fc_grads(forget_grads, "forget_grads")

        prep_time = (time.time() - start_prep_time) / 60

        zu.visualize_fc_grads(forget_grads, "forget_grads")
        if is_diff_grads:
            zu.visualize_fc_grads(retain_grads, "retain_grads")
            zu.visualize_fc_grads(diff_grads, "diff_grads")

        zu.zapping(self.model, weight_mask)

        run_time = 0  # pylint: disable=invalid-name
        start_forget_time = time.time()
        self.model.train()
        for inputs, targets in self.dl["mock_forget"]:
            for input, target in zip(inputs, targets):
                input = input.unsqueeze(0).to(DEVICE, non_blocking=True)
                target = target.unsqueeze(0).to(DEVICE, non_blocking=True)
                self.optimizer.zero_grad()
                output = self.model(input)
                loss = self.loss_fn(output, target)
                loss.backward()
                self.optimizer.step()
        run_time += (time.time() - start_forget_time) / 60  # in minutes
        for epoch in tqdm(range(self.epochs)):
            start_run_time = time.time()
            self.model.train()
            for inputs, targets in self.dl["retain"]:
                inputs = inputs.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)
                # self.optimizer.zero_grad()
                self.model.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()
            epoch_run_time = (time.time() - start_run_time) / 60  # in minutes
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
                    return self.model, epoch, (run_time + prep_time + dl_prep_time)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        return self.model, self.epochs, (run_time + prep_time + dl_prep_time)
