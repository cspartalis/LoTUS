"""
This file contains the implementation of Zap Unlearning
Our proposed method
"""

import copy
import logging
import random
import time

import mlflow
import torch
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import set_config
from eval import (
    compute_accuracy,
    get_membership_attack_data,
    log_membership_attack_prob,
    JSDiv,
)
from seed import set_work_init_fn, set_seed
from unlearning_base_class import UnlearningBaseClass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = set_config()

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


class LinearTemperatureSchedule:
    def __init__(self, min_t: float, max_t: float, max_epochs: int):
        self.min_t = min_t
        self.max_t = max_t
        self.max_epochs = max_epochs

    def __call__(self, epoch: int) -> float:
        return self.min_t + (self.max_t - self.min_t) * (epoch / (self.max_epochs - 1))


class ExponentialTemperatureSchedule:
    def __init__(self, min_t: float, max_t: float, max_epochs: int):
        self.min_t = min_t
        self.max_t = max_t
        self.max_epochs = max_epochs

    def __call__(self, epoch: int) -> float:
        return self.min_t * (self.max_t / self.min_t) ** (epoch / (self.max_epochs - 1))


class SoftmaxWithTemperature(torch.nn.Module):
    def __init__(self):
        super(SoftmaxWithTemperature, self).__init__()

    def forward(self, logits, tau=1):
        return F.softmax(logits / tau, dim=-1)


class GumbelSoftmaxWithTemperature(torch.nn.Module):
    def __init__(self):
        super(GumbelSoftmaxWithTemperature, self).__init__()

    def forward(self, logits, tau=1, hard=False):
        return F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)

    def log(self, logits, tau=1, hard=False, dim=-1):
        """
        Attempt to make log_gumbel_softmax, equivalent to log_softmax.
        This is more numerically stable than taking the log of the output of gumbel_softmax directly.
        """
        # Sample from Gumbel(0, 1)
        gumbels = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )
        gumbels = (logits + gumbels) / tau  # ~Gumbel(logits, tau)

        # Numerically stable log-sum-exp
        y_soft = gumbels - torch.logsumexp(gumbels, dim=dim, keepdim=True)

        if hard:
            # Straight through if hard=True
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(
                logits, memory_format=torch.legacy_contiguous_format
            ).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            ret = y_soft

        return ret


class SAFEMax(UnlearningBaseClass):
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
        self.optimizer = Adam(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        self.teacher = copy.deepcopy(parent_instance.model).to(DEVICE)

        # Set the probability converter (softmax or gumbel-softmax [default])
        if args.probTransformer == "gumbel-softmax":
            self.ProbabilityTranformer = GumbelSoftmaxWithTemperature()
        elif args.probTransformer == "softmax":
            self.ProbabilityTranformer = SoftmaxWithTemperature()
        else:
            raise ValueError("Invalid probability transformer")

        self.beta = args.beta

        set_seed(self.seed, args.cudnn)

        # Logging hyper-parameters
        mlflow.log_param("lr", args.lr)
        mlflow.log_param("wd", args.weight_decay)
        mlflow.log_param("optimizer", self.optimizer)
        mlflow.log_param("tempScheduler", args.tempScheduler)
        mlflow.log_param("probTransformer", args.probTransformer)
        mlflow.log_param("beta", args.beta)
        mlflow.log_param("alpha", args.alpha)

    def unlearn(self, subset_size, is_class_unlearning):
        self.is_class_unlearning = is_class_unlearning
        run_time = 0

        start_prep_time = time.time()

        # Crafting the unlearning dataset (forget set + retain subset).
        indices = list(range(len(self.dl["retain"].dataset)))
        sample_indices = random.sample(
            population=indices,
            k=int(subset_size * len(self.dl["retain"].dataset)),
        )
        retain_subset = torch.utils.data.Subset(
            self.dl["retain"].dataset, sample_indices
        )

        unlearning_data = UnLearningData(
            forget_data=self.dl["forget"].dataset, retain_data=retain_subset
        )

        unlearning_dl = DataLoader(
            unlearning_data,
            batch_size=self.batch_size,
            shuffle=True,
            worker_init_fn=set_work_init_fn(self.seed),
            num_workers=4,
        )

        forget_batch = next(iter(self.dl["forget"]))
        val_batch = next(iter(self.dl["val"]))
        self.teacher.eval()
        with torch.no_grad():
            logits = self.teacher(val_batch[0].to(DEVICE))
            self.teacher_probs = F.softmax(logits, dim=1).to("cpu")
        # Compute class indices for both teacher and student probs
        self.t_class_idx = torch.argmax(self.teacher_probs, dim=1)

        prep_time = (time.time() - start_prep_time) / 60  # in minutes

        # Unlearnig loop
        for epoch in tqdm(range(self.epochs)):
            start_epoch_time = time.time()

            # Compute JS_div
            self.model.eval()
            with torch.no_grad():
                logits = self.model(forget_batch[0].to(DEVICE))
                student_probs = F.softmax(logits, dim=1).to("cpu")
            s_class_idx = torch.argmax(student_probs, dim=1)

            list_JSD = self.compute_JS_div(s_class_idx, student_probs)
            avg_JSD = np.mean(list_JSD)
            mlflow.log_metric("JSD", avg_JSD, step=(epoch + 1))
            # Early stopping
            if avg_JSD < 1e-6:
                break

            self.temperature = np.exp(args.alpha * avg_JSD)

            self.model.train()

            for x, y, l in unlearning_dl:
                x = x.to(DEVICE)
                l = l.to(DEVICE)  # 1 if the sample is from the forget set, 0 otherwise
                with torch.no_grad():
                    teacher_logits = self.teacher(x)
                output = self.model(x)
                self.optimizer.zero_grad()
                loss = self.unlearning_loss(output, l, teacher_logits)
                loss.backward()
                self.optimizer.step()

            epoch_run_time = (time.time() - start_epoch_time) / 60  # in minutes
            run_time += epoch_run_time

            # acc_retain = compute_accuracy(self.model, self.dl["retain"], False)
            # acc_forget = compute_accuracy(self.model, self.dl["forget"], False)

            # # Log accuracies
            # mlflow.log_metric("acc_retain", acc_retain, step=(epoch + 1))
            # mlflow.log_metric("acc_forget", acc_forget, step=(epoch + 1))

        return self.model, run_time + prep_time

    def unlearning_loss(self, outputs, l, teacher_logits):
        """
        args:
            outputs: output of the unlearned/student model
            l: 1 if the sample is from the forget set, 0 otherwise
            full_teacher_logits: logits of the teacher model
            teacher_logits: logits of the original/teacher model.
            temperature: temperature for gumbel-softmax annealing
            beta: Regularization coefficient penalizing agreement between student and teacher argmax
        returns:
            mean_loss: mean loss over the batch
        """
        l = torch.unsqueeze(l, dim=1)
        retain_probs = self.ProbabilityTranformer(teacher_logits, hard=True)
        forget_probs = self.ProbabilityTranformer(teacher_logits, tau=self.temperature)
        t = l * forget_probs + (1 - l) * retain_probs  # Teacher prob dist
        hard_t = l * self.ProbabilityTranformer(teacher_logits, hard=True)
        log_s = F.log_softmax(outputs, dim=1)
        s = F.softmax(outputs, dim=1)

        loss = torch.sum(-(t * log_s) + self.beta * l * (hard_t * s), dim=1)

        mean_loss = torch.mean(loss)

        return mean_loss

    def compute_JS_div(self, s_class_idx, student_probs):
        list_JSD = []
        for i in range(len(self.t_class_idx)):
            for j in range(len(s_class_idx)):
                if self.t_class_idx[i] == s_class_idx[j]:
                    JSD = JSDiv(self.teacher_probs[i], student_probs[j])
                    list_JSD.append(JSD)
        return list_JSD
