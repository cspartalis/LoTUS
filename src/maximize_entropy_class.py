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
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import set_config
from eval import (
    compute_accuracy,
    get_membership_attack_data,
    log_membership_attack_prob,
)
from seed import set_work_init_fn
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
        self.optimizer = Adam(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        self.teacher = copy.deepcopy(parent_instance.model).to(DEVICE)
        mlflow.log_param("lr", args.lr)
        mlflow.log_param("wd", args.weight_decay)
        mlflow.log_param("optimizer", self.optimizer)

    def unlearn(self, subset_size):

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

        self.teacher.eval()

        prep_time = (time.time() - start_prep_time) / 60  # in minutes

        # Unlearnig loop
        for epoch in tqdm(range(self.epochs)):
            start_epoch_time = time.time()
            self.model.train()

            for x, y, l in unlearning_dl:
                x = x.to(DEVICE)
                l = l.to(DEVICE)
                with torch.no_grad():
                    teacher_logits = self.teacher(x)
                output = self.model(x)
                self.optimizer.zero_grad()
                loss = self.unlearning_loss(output, y, l, teacher_logits, epoch)
                loss.backward()
                self.optimizer.step()

            epoch_run_time = (time.time() - start_epoch_time) / 60  # in minutes
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

        return self.model, run_time + prep_time

    def unlearning_loss(self, outputs, targets, labels, teacher_logits, epochs):
        """
        args:
            outputs: output of the unlearned/student model
            targets: ground truth labels
            labels: 1 if the sample is from the forget set, 0 otherwise
            full_teacher_logits: logits of the teacher model
            teacher_logits: logits of the original/teacher model.
            epochs: number of epochs to consider for the top-k values
        returns:
            mean_loss: mean loss over the batch
        """
        labels = torch.unsqueeze(labels, dim=1)

        teacher_out = F.softmax(teacher_logits, dim=1)
        thinkHarder_out = teacher_out.clone().detach().cpu()

        # Find the indices and values of the target labels in the batch.
        row_indices = torch.arange(thinkHarder_out.shape[0])
        selected_values = thinkHarder_out[row_indices, targets]
        # Calculate the probability to be shared among the other classes.
        prob_to_dist = selected_values / (self.num_classes - 1)
        thinkHarder_out += prob_to_dist.unsqueeze(1)
        # The ground truth labels are always assigned a probability of 0.
        thinkHarder_out.scatter_(1, targets.unsqueeze(1), 0)

        # If you are in the 2nd iteration, assign 0 probability to the ground truth label,
        # assign 1/(num_classes - 1) to highest probability (apart from the ground truth label),
        # and distribute the remaining probability among the other classes.
        # If you are in the 3rd iteration, assign 0 probability to the ground truth label,
        # assign 1/(num_classes - 1) to the top-2 highest probabilities (apart from the ground truth label),
        # and distribute the remaining probability among the other classes etc...
        if epochs > 0 and epochs < self.num_classes - 1:
            topk_values, topk_indices = torch.topk(thinkHarder_out, epochs, dim=1)
            prob_to_dist = torch.sum(topk_values, dim=1) - (
                epochs / (self.num_classes - 1)
            )
            thinkHarder_out += prob_to_dist.unsqueeze(1)
            thinkHarder_out.scatter_(1, topk_indices, 1 / (self.num_classes - 1))
            thinkHarder_out.scatter_(1, targets.unsqueeze(1), 0)
        # If iterations > num_classes, assign 1/(num_classes - 1) to all classes
        # except the ground truth label (which is assigned 0 probability).
        elif epochs > self.num_classes - 1:
            thinkHarder_out = (
                torch.ones_like(thinkHarder_out) * 1 / (self.num_classes - 1)
            )
            thinkHarder_out.scatter_(1, targets.unsqueeze(1), 0)

        thinkHarder_out = thinkHarder_out.to(DEVICE)

        overall_out = labels * thinkHarder_out + (1 - labels) * teacher_out
        student_out = F.log_softmax(outputs, dim=1)

        cross_entropy_loss = -torch.sum(overall_out * student_out, dim=1)
        mean_loss = torch.mean(cross_entropy_loss)

        return mean_loss
