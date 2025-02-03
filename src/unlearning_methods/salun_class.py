import copy
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import random
import time
import mlflow
import numpy as np
from tqdm import tqdm
from torch.nn.functional import log_softmax, gumbel_softmax
from helpers.seed import set_work_init_fn
from unlearning_methods.unlearning_base_class import UnlearningBaseClass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SalUn(UnlearningBaseClass):
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
        self.lr = 0.1
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        self.criterion = nn.CrossEntropyLoss()
        self.masks_dir = os.path.join(os.path.dirname(__file__), "salun_masks")

    def save_gradient_ratio(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        gradients = {}

        forget_loader = self.dl["forget"]
        self.model.eval()

        for name, param in self.model.named_parameters():
            gradients[name] = 0

        for i, (image, target) in enumerate(forget_loader):
            image = image.cuda()
            target = target.cuda()

            # compute output
            output_clean = self.model(image)
            loss = -self.criterion(output_clean, target)

            optimizer.zero_grad()
            loss.backward()

            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        gradients[name] += param.grad.data

        with torch.no_grad():
            for name in gradients:
                gradients[name] = torch.abs_(gradients[name])

        threshold_list = [0.1]

        for i in threshold_list:
            sorted_dict_positions = {}
            hard_dict = {}

            # Concatenate all tensors into a single tensor
            all_elements = -torch.cat(
                [tensor.flatten() for tensor in gradients.values()]
            )

            # Calculate the threshold index for the top 10% elements
            threshold_index = int(len(all_elements) * i)

            # Calculate positions of all elements
            positions = torch.argsort(all_elements)
            ranks = torch.argsort(positions)

            start_index = 0
            for key, tensor in gradients.items():
                num_elements = tensor.numel()
                # tensor_positions = positions[start_index: start_index + num_elements]
                tensor_ranks = ranks[start_index : start_index + num_elements]

                sorted_positions = tensor_ranks.reshape(tensor.shape)
                sorted_dict_positions[key] = sorted_positions

                # Set the corresponding elements to 1
                threshold_tensor = torch.zeros_like(tensor_ranks)
                threshold_tensor[tensor_ranks < threshold_index] = 1
                threshold_tensor = threshold_tensor.reshape(tensor.shape)
                hard_dict[key] = threshold_tensor
                start_index += num_elements

            torch.save(hard_dict, os.path.join(self.masks_dir, "with_{}.pt".format(i)))

    def generate_mask(self):
        os.makedirs(self.masks_dir, exist_ok=True)
        self.save_gradient_ratio()

    def unlearn(self, unlearning_method, is_class_unlearning):
        start_time = time.time()
        self.generate_mask()
        mask = torch.load(os.path.join(self.masks_dir, "with_0.1.pt"))
        prep_time = time.time() - start_time
        self.model.zero_grad()
        if unlearning_method == "relabel":
            model, run_time = self.salun_with_relabel(mask)
        elif unlearning_method == "lotus":
            model, run_time = self.salun_with_lotus(mask, is_class_unlearning)
        return model, prep_time + run_time

    def salun_with_relabel(self, mask):
        lr = 1e-3
        momentum = 0.9
        weight_decay = 5e-4
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        loss_fn = torch.nn.CrossEntropyLoss()

        run_time = 0

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

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()

                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]

                optimizer.step()

            # Repair stage: Finetune on the "retain" set
            for inputs, targets in self.dl["retain"]:
                inputs = inputs.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()

            epoch_run_time = (time.time() - start_time) / 60  # in minutes
            run_time += epoch_run_time

        return self.model, run_time

    def salun_with_lotus(self, mask, is_class_unlearning):
        from unlearning_methods.lotus_class import UnLearningData
        from helpers.eval import compute_accuracy

        subset_size = 0.3
        is_class_unlearning = False
        alpha = 2
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=5e-4,
        )
        teacher = copy.deepcopy(self.model)

        run_time = 0
        start_prep_time = time.time()
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

        unlearning_dl = torch.utils.data.DataLoader(
            unlearning_data,
            batch_size=self.batch_size,
            shuffle=True,
            worker_init_fn=set_work_init_fn(self.seed),
            num_workers=4,
            pin_memory=True,
        )

        teacher.eval()
        if is_class_unlearning:
            acc_val_t = 0
        else:
            acc_val_t = compute_accuracy(teacher, self.dl["val"])
        prep_time = (time.time() - start_prep_time) / 60  # in minutes

        for epoch in tqdm(range(self.epochs)):
            start_epoch_time = time.time()

            self.model.eval()
            acc_forget_s = compute_accuracy(self.model, self.dl["forget"])
            acc_diff = acc_forget_s - acc_val_t
            if acc_diff <= -0.01:
                break
            temperature = torch.tensor(np.exp(alpha * acc_diff), device=DEVICE)

            self.model.train()

            for x, y, l in unlearning_dl:
                x = x.to(DEVICE, non_blocking=True)
                l = l.to(DEVICE, non_blocking=True)
                with torch.no_grad():
                    teacher_logits = teacher(x)
                output = self.model(x)
                # del x, y
                optimizer.zero_grad()
                loss = self.unlearning_loss(output, l, teacher_logits, temperature)
                # del output, l, teacher_logits
                loss.backward()
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]

        
                optimizer.step()
                # torch.cuda.empty_cache()

            epoch_run_time = (time.time() - start_epoch_time) / 60  # in minutes
            run_time += epoch_run_time

        self.model.eval()
        # acc_retain = compute_accuracy(self.model, self.dl["retain"])
        # mlflow.log_metric("acc_forget", acc_forget_s, step=(epoch + 1))
        # mlflow.log_metric("acc_retain", acc_retain, step=(epoch + 1))
        # del self.teacher
        # torch.cuda.empty_cache()
        return self.model, run_time + prep_time

    def unlearning_loss(self, outputs, l, teacher_logits, temperature):
        """
        Computes the unlearning loss for the given outputs and teacher logits.
        Args:
            outputs (torch.Tensor): The output logits from the model.
            l (torch.Tensor): A tensor indicating the if the data point needs to be
                              unlearned (l=1) or retained (l=0).
            teacher_logits (torch.Tensor): The logits from the teacher model.
        Returns:
            torch.Tensor: The mean unlearning loss.
        """

        l = torch.unsqueeze(l, dim=1)
        hard_t = gumbel_softmax(teacher_logits, hard=True)
        soft_t = gumbel_softmax(teacher_logits, tau=temperature, hard=False)
        t = l * soft_t + (1 - l) * hard_t  # Teacher prob dist
        log_s = log_softmax(outputs, dim=1)

        loss = -torch.sum(t * log_s, dim=1)
        return loss.mean()
