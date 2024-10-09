"""
Paper: Can Bad Teaching Induce Unlearning? by Chundawat et al. (2023)
Code: https://tinyurl.com/5e7xxb3p 
Hyper-parameters:
* KL_temperature = 1
* optimizer = torch.optim.Adam(student_model.parameters(), lr=0.0001)
"""

import copy
import random
import time

import mlflow
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from seed import set_work_init_fn
from unlearning_base_class import UnlearningBaseClass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
            y = 1
            return x, y
        else:
            x = self.retain_data[index - self.forget_len][0]
            y = 0
            return x, y


class bad_teachingUnlearning(UnlearningBaseClass):
    def __init__(self, parent_instance, unlearning_teacher):
        super().__init__(
            parent_instance.dl,
            parent_instance.batch_size,
            parent_instance.num_classes,
            parent_instance.model,
            parent_instance.epochs,
            parent_instance.dataset,
            parent_instance.seed,
        )
        self.unlearning_teacher = copy.deepcopy(unlearning_teacher).to(DEVICE)
        self.full_trained_teacher = copy.deepcopy(parent_instance.model).to(DEVICE)
        self.KL_temperature = 1

        self.is_multi_label = True if parent_instance.dataset == "mucac" else False
        if self.is_multi_label:
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss()
        self.lr = 0.0001
        self.weight_decay = 0
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        mlflow.log_param("loss", "cross_entropy")
        mlflow.log_param("lr", self.lr)
        mlflow.log_param("weight_decay", self.weight_decay)

    def UnlearnerLoss(
        self, output, labels, full_teacher_logits, unlearn_teacher_logits
    ):
        labels = torch.unsqueeze(labels, dim=1)

        f_teacher_out = F.softmax(full_teacher_logits / self.KL_temperature, dim=1)
        u_teacher_out = F.softmax(unlearn_teacher_logits / self.KL_temperature, dim=1)

        # label 1 means forget sample
        # label 0 means retain sample
        overall_teacher_out = labels * u_teacher_out + (1 - labels) * f_teacher_out
        student_out = F.log_softmax(output / self.KL_temperature, dim=1)
        return F.kl_div(student_out, overall_teacher_out)

    def unlearn(self):
        run_time = 0  # pylint: disable=invalid-name

        start_dl_prep_time = time.time()

        # creating the unlearning dataset.
        indices = list(range(len(self.dl["retain"].dataset)))
        sample_indices = random.sample(
            population=indices,
            k=int(0.3 * len(self.dl["retain"].dataset)),
        )

        retain_train_subset = torch.utils.data.Subset(
            self.dl["retain"].dataset, sample_indices
        )

        unlearning_data = UnLearningData(
            forget_data=self.dl["forget"].dataset, retain_data=retain_train_subset
        )
        unlearning_loader = DataLoader(
            unlearning_data,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            worker_init_fn=set_work_init_fn(self.seed),
            num_workers=4,
        )
        self.unlearning_teacher.eval()
        self.full_trained_teacher.eval()
        dl_prep_time = time.time() - start_dl_prep_time

        for epoch in tqdm(range(self.epochs)):
            start_time = time.time()
            self.model.train()
            for batch in unlearning_loader:
                x, y = batch
                x, y = x.to(DEVICE), y.to(DEVICE)
                with torch.no_grad():
                    full_teacher_logits = self.full_trained_teacher(x)
                    unlearn_teacher_logits = self.unlearning_teacher(x)
                output = self.model(x)
                self.optimizer.zero_grad()
                loss = self.UnlearnerLoss(
                    output=output,
                    labels=y,
                    full_teacher_logits=full_teacher_logits,
                    unlearn_teacher_logits=unlearn_teacher_logits,
                )
                loss.backward()
                self.optimizer.step()
            epoch_run_time = (time.time() - start_time) / 60  # in minutes
            run_time += epoch_run_time

        return self.model, run_time + dl_prep_time
