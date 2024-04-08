"""
This file contains the implementation of UNSIR Unlearning:
https://arxiv.org/pdf/2111.08947.pdf
"""

import os
import time

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from tqdm import tqdm

from eval import compute_accuracy
from seed import set_work_init_fn
from unlearning_base_class import UnlearningBaseClass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Noise(nn.Module):
    def __init__(self, batch_size, *dim):
        super().__init__()
        self.noise = nn.Parameter(torch.randn(batch_size, *dim), requires_grad=True)

    def forward(self):
        return self.noise


def float_to_uint8(img_float):
    """Convert a floating point image in the range [0,1] to uint8 image in the range [0,255]."""
    img_uint8 = (img_float * 255).astype(np.uint8)
    return img_uint8


class UNSIR(UnlearningBaseClass):
    """
    Adapted from: https://github.com/vikram2000b/Fast-Machine-Unlearning/blob/main/Machine%20Unlearning.ipynb
    Modifications for instance-wise unlearning from:
    https://github.com/ndb796/MachineUnlearning/blob/main/01_MUFAC/Machine_Unlearning_MUFAC_UNSIR.ipynb

    Impair phase (Stage 1): Update noise to increase the distance between the model and the forget dataset.
    The updated noise is integrated into the trainind dataset to enhance the model's ability to forget the
    specific dataset
    Repair phase (Stage 2): Repair the impaird model using the retain dataset
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
        self.lr = 1e-3
        self.momentum = 0.9
        self.weight_decay = 0
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        self.lr_scheduler = None

        mlflow.log_param("loss", self.loss_fn)
        mlflow.log_param("lr", self.lr)
        mlflow.log_param("momentum", self.momentum)
        mlflow.log_param("weight_decay", self.weight_decay)
        mlflow.log_param("optimizer", self.optimizer)
        mlflow.log_param("lr_scheduler", "None")

    def unlearn(self, seed):
        run_time = 0  # pylint: disable=invalid-name

        prep_time_start = time.time()
        # Start of my snippet of code

        # Create a subset of the retain dataset with maximum 1000 samples per class (as denoted in the UNSIR repo)
        if self.dataset == "cifar-100":
            max_retain_samples_per_class = 100
        else:
            max_retain_samples_per_class = 1000

        subset_retain_dataset = []
        class_counts = {}
        if self.is_multi_label == False:
            for x_retain, y_retain in self.dl["retain"].dataset:
                if isinstance(y_retain, torch.Tensor):
                    y_retain = y_retain.item()
                class_label = y_retain
                if class_label not in class_counts:
                    class_counts[class_label] = 0
                if class_counts[class_label] < max_retain_samples_per_class:
                    subset_retain_dataset.append((x_retain, y_retain))
                    class_counts[class_label] += 1
                if (
                    len(subset_retain_dataset)
                    >= max_retain_samples_per_class * self.num_classes
                ):
                    break
        else:
            # Using the dl["retain"].dataset, get 3000 samples ramdomly
            subset_retain_dataset = list(Subset(self.dl["retain"].dataset, range(3000)))

        subset_retain_dataloader = torch.utils.data.DataLoader(
            subset_retain_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            worker_init_fn=set_work_init_fn(seed),
        )

        prep_time = (time.time() - prep_time_start) / 60  # in minutes
        # End of my snippet of code

        for epoch in tqdm(range(self.epochs)):
            start_time = time.time()

            # Step 1: Impair
            for batch_idx, ((x_retain, y_retain), (x_forget, y_forget)) in enumerate(
                zip(subset_retain_dataloader, self.dl["forget"])
            ):
                y_retain = y_retain.to(DEVICE)
                batch_size_forget = self.batch_size

                if (
                    x_retain.size(0) != self.batch_size
                    or x_forget.size(0) != self.batch_size
                ):
                    continue

                # Initialize the noise.
                noise_dim = x_retain.size(1), x_retain.size(2), x_retain.size(3)
                noise = Noise(batch_size_forget, *noise_dim).cuda()
                noise_optimizer = torch.optim.Adam(noise.parameters(), lr=0.01)
                noise_tensor = noise()[:batch_size_forget]

                # Update the noise for increasing the loss value.
                for _ in range(5):
                    outputs = self.model(noise_tensor)
                    with torch.no_grad():
                        target_logits = self.model(x_forget.cuda())
                    # Maximize the similarity between noise data and forget features.
                    loss_noise = -F.mse_loss(outputs, target_logits)

                # Backpropagate to update the noise.
                noise_optimizer.zero_grad()
                loss_noise.backward(retain_graph=True)
                noise_optimizer.step()

                self.model.train()
                noise_tensor = torch.clamp(noise_tensor, 0, 1).detach().to(DEVICE)
                outputs = self.model(noise_tensor)
                loss1 = self.loss_fn(outputs, y_retain)

                outputs = self.model(x_retain.to(DEVICE))
                loss2 = self.loss_fn(outputs, y_retain)

                joint_loss = loss1 + loss2

                self.optimizer.zero_grad()
                joint_loss.backward()
                self.optimizer.step()

            # Step 2: Repair
            for batch_idx, (x_retain, y_retain) in enumerate(subset_retain_dataloader):
                y_retain = y_retain.to(DEVICE)

                # Classification Loss
                outputs_retain = self.model(x_retain.to(DEVICE))
                classification_loss = self.loss_fn(outputs_retain, y_retain)

                self.optimizer.zero_grad()
                classification_loss.backward()
                self.optimizer.step()

            epoch_run_time = (time.time() - start_time) / 60  # in minutes
            run_time += epoch_run_time

            acc_retain = compute_accuracy(
                self.model, self.dl["retain"], self.is_multi_label
            )
            acc_forget = compute_accuracy(
                self.model, self.dl["forget"], self.is_multi_label
            )
            acc_val = compute_accuracy(self.model, self.dl["val"], self.is_multi_label)

            # Log accuracies
            mlflow.log_metric("acc_retain", acc_retain, step=(epoch + 1))
            mlflow.log_metric("acc_val", acc_val, step=(epoch + 1))
            mlflow.log_metric("acc_forget", acc_forget, step=(epoch + 1))

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        return self.model, run_time + prep_time
