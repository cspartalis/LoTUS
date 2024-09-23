import copy
import logging
import random
import time

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import set_config
from eval import compute_accuracy
from seed import set_seed, set_work_init_fn
from unlearning_base_class import UnlearningBaseClass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = set_config()

log = logging.getLogger(__name__)


class UnLearningData(Dataset):
    """
    Code from: "Can Bad Teaching induce unleanring?"
    """

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


class SoftmaxWithTemperature(torch.nn.Module):
    def __init__(self):
        super(SoftmaxWithTemperature, self).__init__()

    def forward(self, logits, tau=1, hard=False):
        if hard==True:
            return F.one_hot(torch.argmax(logits, dim=-1), num_classes=logits.shape[-1])
        else:
            return F.softmax(logits / tau, dim=-1)


class GumbelSoftmaxWithTemperature(torch.nn.Module):
    def __init__(self):
        super(GumbelSoftmaxWithTemperature, self).__init__()

    def forward(self, logits, tau=1, hard=False):
        return F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)


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
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        self.teacher = copy.deepcopy(parent_instance.model).to(DEVICE)

        if args.probTransformer == "gumbel-softmax":
            self.ProbabilityTranformer = GumbelSoftmaxWithTemperature()
        elif args.probTransformer == "softmax":
            self.ProbabilityTranformer = SoftmaxWithTemperature()
        else:
            raise ValueError("Invalid probability transformer")

        self.alpha = args.alpha
        self.beta = args.beta

        set_seed(self.seed, args.cudnn)

        # Logging hyper-parameters
        mlflow.log_param("lr", args.lr)
        mlflow.log_param("wd", args.weight_decay)
        mlflow.log_param("optimizer", self.optimizer)
        mlflow.log_param("tempScheduler", args.tempScheduler)
        mlflow.log_param("probTransformer", args.probTransformer)
        mlflow.log_param("beta", self.beta)
        mlflow.log_param("alpha", self.alpha)

    def unlearning_loss(self, outputs, l, teacher_logits):
        l = torch.unsqueeze(l, dim=1)
        hard_t = self.ProbabilityTranformer(teacher_logits, hard=True)
        soft_t = self.ProbabilityTranformer(teacher_logits, tau=self.temperature)
        t = l * soft_t + (1 - l) * hard_t  # Teacher prob dist
        log_s = F.log_softmax(outputs, dim=1)

        loss = -torch.sum(t * log_s, dim=1) + self.beta * l.squeeze() * torch.sum(
            hard_t * torch.exp(log_s), dim=1
        )

        return loss.mean()

    def unlearn(self, subset_size, is_class_unlearning):
        self.is_class_unlearning = is_class_unlearning

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
            pin_memory=True,
        )

        run_time = 0
        start_prep_time = time.time()

        self.teacher.eval()
        acc_val_t = compute_accuracy(self.teacher, self.dl["val"], False)
        prep_time = (time.time() - start_prep_time) / 60  # in minutes

        mlflow.log_metric("prep_time", prep_time)

        for epoch in tqdm(range(self.epochs)):
            start_epoch_time = time.time()

            self.model.eval()
            acc_forget_s = compute_accuracy(self.model, self.dl["forget"], False)
            acc_diff = acc_forget_s - acc_val_t
            if acc_diff <= -0.03:
                break
            self.temperature = torch.tensor(
                np.exp(self.alpha * acc_diff), device=DEVICE
            )

            self.model.train()

            for x, y, l in unlearning_dl:
                x = x.to(DEVICE, non_blocking=True)
                l = l.to(DEVICE, non_blocking=True)
                with torch.no_grad():
                    teacher_logits = self.teacher(x)
                output = self.model(x)
                self.optimizer.zero_grad()
                loss = self.unlearning_loss(output, l, teacher_logits)
                loss.backward()
                self.optimizer.step()

            epoch_run_time = (time.time() - start_epoch_time) / 60  # in minutes
            run_time += epoch_run_time

            acc_retain = compute_accuracy(self.model, self.dl["retain"], False)
            mlflow.log_metric("acc_forget", acc_forget_s, step=(epoch + 1))
            mlflow.log_metric("acc_retain", acc_retain, step=(epoch + 1))

        return self.model, run_time + prep_time
