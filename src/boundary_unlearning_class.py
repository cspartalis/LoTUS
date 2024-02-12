"""
This file contains the implementation of Boundary Unlearning
https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_Boundary_Unlearning_Rapid_Forgetting_of_Deep_Networks_via_Shifting_the_CVPR_2023_paper.pdf
"""

import copy
import time

import mlflow
import torch
import torch.nn as nn
from tqdm import tqdm

from eval import compute_accuracy
from unlearning_base_class import UnlearningBaseClass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BoundaryUnlearning(UnlearningBaseClass):
    """
    Code from the paper "Boundary Unlearning".
    https://www.dropbox.com/s/bwu543qsdy4s32i/Boundary-Unlearning-Code.zip?dl=0
    All the var values are the same as in the original codebase.
    Whereverer there are snippets of my code, I have added a corresponding comment.
    The arguments that should be passed to the function to match the paper are:
    - loss = cross_entropy
    - optimizer = sgd
    - lr = 1e-5
    - momentum = 0.9
    - weight_decay = 0
    - lr_scheduler = None
    """

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
        self.lr = 1e-5
        self.momentum = 0.9
        self.weight_decay = 0
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

    def unlearn(self):
        # start of my snippet
        start_prep_time = time.time()
        # end of my snippet

        test_model = copy.deepcopy(self.model).to(DEVICE)
        unlearn_model = copy.deepcopy(self.model).to(DEVICE)

        adv = FGSM(test_model, bound=0.1, norm=True, random_start=False, device=DEVICE)
        forget_data_gen = inf_generator(self.dl["forget"])
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

            acc_retain = compute_accuracy(self.model, self.dl["retain"], self.is_multi_label)
            acc_forget = compute_accuracy(self.model, self.dl["forget"], self.is_multi_label)
            acc_val = compute_accuracy(self.model, self.dl["val"], self.is_multi_label)

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

        return self.model, run_time + prep_time


class AttackBase(object):
    def __init__(self, model=None, norm=False, discrete=True, device=None):
        self.model = model
        self.norm = norm
        # Normalization are needed for CIFAR10, ImageNet
        if self.norm:
            self.mean = (0.4914, 0.4822, 0.2265)
            self.std = (0.2023, 0.1994, 0.2010)
        self.discrete = discrete
        self.device = device or torch.device("cuda:0")
        self.loss(device=self.device)

    def loss(self, custom_loss=None, device=None):
        device = device or self.device
        self.criterion = custom_loss or nn.CrossEntropyLoss()
        self.criterion.to(device)

    def perturb(self, x):
        raise NotImplementedError

    def normalize(self, x):
        if self.norm:
            y = x.clone().to(x.device)
            num_channels = y.shape[1]
            for i in range(num_channels):
                # start of my snippet (not it works independently of the number of channels)
                y[:, i, :, :] = (y[:, i, :, :] - self.mean[i]) / self.std[i]
                # end of my snippet
            # Commented out these 3 lines from the original codebase
            # y[:, 0, :, :] = (y[:, 0, :, :] - self.mean[0]) / self.std[0]
            # y[:, 1, :, :] = (y[:, 1, :, :] - self.mean[1]) / self.std[1]
            # y[:, 2, :, :] = (y[:, 2, :, :] - self.mean[2]) / self.std[2]
            return y
        return x

    def inverse_normalize(self, x):
        if self.norm:
            y = x.clone().to(x.device)
            num_channels = y.shape[1]
            for i in range(num_channels):
                # start of my snippet (not it works independently of the number of channels)
                y[:, i, :, :] = y[:, i, :, :] * self.std[i] + self.mean[i]
                # end of my snippet
            # Commented out these 3 lines from the original codebase
            # y[:, 0, :, :] = y[:, 0, :, :] * self.std[0] + self.mean[0]
            # y[:, 1, :, :] = y[:, 1, :, :] * self.std[1] + self.mean[1]
            # y[:, 2, :, :] = y[:, 2, :, :] * self.std[2] + self.mean[2]
            return y
        return x

    def discretize(self, x):
        return torch.round(x * 255) / 255

    # Change this name as "projection"
    def clamper(self, x_adv, x_nat, bound=None, metric="inf", inverse_normalized=False):
        if not inverse_normalized:
            x_adv = self.inverse_normalize(x_adv)
            x_nat = self.inverse_normalize(x_nat)
        if metric == "inf":
            clamp_delta = torch.clamp(x_adv - x_nat, -bound, bound)
        else:
            clamp_delta = x_adv - x_nat
            for batch_index in range(clamp_delta.size(0)):
                image_delta = clamp_delta[batch_index]
                image_norm = image_delta.norm(p=metric, keepdim=False)
                # TODO: channel isolation?
                if image_norm > bound:
                    clamp_delta[batch_index] /= image_norm
                    clamp_delta[batch_index] *= bound
        x_adv = x_nat + clamp_delta
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        return (
            self.normalize(self.discretize(x_adv)).clone().detach().requires_grad_(True)
        )


class FGSM(AttackBase):
    def __init__(
        self,
        model=None,
        bound=None,
        norm=False,
        random_start=False,
        discrete=True,
        device=None,
        **kwargs,
    ):
        super(FGSM, self).__init__(model, norm, discrete, device)
        self.bound = bound
        self.rand = random_start

    # @overrides
    def perturb(self, x, y, model=None, bound=None, device=None, **kwargs):
        criterion = self.criterion
        model = model or self.model
        bound = bound or self.bound
        device = device or self.device

        model.zero_grad()
        x_nat = self.inverse_normalize(x.detach().clone().to(device))
        x_adv = x.detach().clone().requires_grad_(True).to(device)
        if self.rand:
            rand_perturb_dist = distributions.uniform.Uniform(-bound, bound)
            rand_perturb = rand_perturb_dist.sample(sample_shape=x_adv.shape).to(device)
            x_adv = self.clamper(
                self.inverse_normalize(x_adv) + rand_perturb,
                x_nat,
                bound=bound,
                inverse_normalized=True,
            )
            if self.discretize:
                x_adv = (
                    self.normalize(self.discretize(x_adv))
                    .detach()
                    .clone()
                    .requires_grad_(True)
                )
            else:
                x_adv = self.normalize(x_adv).detach().clone().requires_grad_(True)

        pred = model(x_adv)
        if criterion.__class__.__name__ == "NLLLoss":
            pred = F.softmax(pred, dim=-1)
        loss = criterion(pred, y)
        loss.backward()

        grad_sign = x_adv.grad.data.detach().sign()
        x_adv = self.inverse_normalize(x_adv) + grad_sign * bound
        x_adv = self.clamper(x_adv, x_nat, bound=bound, inverse_normalized=True)

        return x_adv.detach()


def inf_generator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()
