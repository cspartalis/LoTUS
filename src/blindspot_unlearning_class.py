""" https://github.com/if-loops/selective-synaptic-dampening/blob/27bcab379aa74ad046e876a694761d3d4b3a3600/src/unlearn.py """

import copy
import random
import time

import mlflow
import torch
from torch.nn import functional as F
from tqdm import tqdm

from eval import compute_accuracy
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

class BlindspotUnlearning(UnlearningBaseClass):
    def __init__(self, parent_instance, unlearning_teacher):
        super().__init__(
            parent_instance.dl,
            parent_instance.batch_size,
            parent_instance.num_classes,
            parent_instance.model,
            parent_instance.epochs,
            parent_instance.dataset,
        )
        self.unlearning_teacher = copy.deepcopy(unlearning_teacher)
        self.full_trained_teacher = copy.deepcopy(parent_instance.model)
        self.KL_temperature = 1

        self.is_multi_label = True if parent_instance.dataset == "mucac" else False
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
        # mlflow.log_param("momentum", self.momentum)
        mlflow.log_param("weight_decay", self.weight_decay)
        mlflow.log_param("optimizer", self.optimizer)
        mlflow.log_param("lr_scheduler", "None")

   def UnlearnerLoss(
        output, labels, full_teacher_logits, unlearn_teacher_logits, KL_temperature
    ):
        labels = torch.unsqueeze(labels, dim=1)

        f_teacher_out = F.softmax(full_teacher_logits / KL_temperature, dim=1)
        u_teacher_out = F.softmax(unlearn_teacher_logits / KL_temperature, dim=1)

        # label 1 means forget sample
        # label 0 means retain sample
        overall_teacher_out = labels * u_teacher_out + (1 - labels) * f_teacher_out
        student_out = F.log_softmax(output / KL_temperature, dim=1)
        return F.kl_div(student_out, overall_teacher_out)


    def unlearning_step(
        model,
        unlearning_teacher,
        full_trained_teacher,
        unlearn_data_loader,
        optimizer,
        device,
        KL_temperature,
    ):
        losses = []
        for batch in unlearn_data_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                full_teacher_logits = full_trained_teacher(x)
                unlearn_teacher_logits = unlearning_teacher(x)
            output = model(x)
            optimizer.zero_grad()
            loss = UnlearnerLoss(
                output=output,
                labels=y,
                full_teacher_logits=full_teacher_logits,
                unlearn_teacher_logits=unlearn_teacher_logits,
                KL_temperature=KL_temperature,
            )
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu().numpy())
        return np.mean(losses)


    def fit_one_unlearning_cycle(epochs, model, train_loader, val_loader, lr, device):
        history = []

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            model.train()
            train_losses = []
            lrs = []
            for batch in train_loader:
                loss = training_step(model, batch, device)
                loss.backward()
                train_losses.append(loss.detach().cpu())

                optimizer.step()
                optimizer.zero_grad()

                lrs.append(get_lr(optimizer))

            result = evaluate(model, val_loader, device)
            result["train_loss"] = torch.stack(train_losses).mean()
            result["lrs"] = lrs
            epoch_end(model, epoch, result)
            history.append(result)
        return history


    def blindspot_unlearner(
        model,
        unlearning_teacher,
        full_trained_teacher,
        retain_data,
        forget_data,
        epochs=10,
        optimizer="adam",
        lr=0.01,
        batch_size=256,
        device="cuda",
        KL_temperature=1,
    ):
        # creating the unlearning dataset.
        unlearning_data = UnLearningData(forget_data=forget_data, retain_data=retain_data)
        unlearning_loader = DataLoader(
            unlearning_data, batch_size=batch_size, shuffle=True, pin_memory=True
        )

        unlearning_teacher.eval()
        full_trained_teacher.eval()
        optimizer = optimizer
        if optimizer == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            # if optimizer is not a valid string, then assuming it as a function to return optimizer
            optimizer = optimizer  # (model.parameters())

        for epoch in range(epochs):
            loss = unlearning_step(
                model=model,
                unlearning_teacher=unlearning_teacher,
                full_trained_teacher=full_trained_teacher,
                unlearn_data_loader=unlearning_loader,
                optimizer=optimizer,
                device=device,
                KL_temperature=KL_temperature,
            )
            print("Epoch {} Unlearning Loss {}".format(epoch + 1, loss))

    def blindspot(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
        **kwargs,
    ):
        student_model = deepcopy(model)
        KL_temperature = 1
        optimizer = torch.optim.Adam(student_model.parameters(), lr=0.0001)
        retain_train_subset = random.sample(
            retain_train_dl.dataset, int(0.3 * len(retain_train_dl.dataset))
        )

        if kwargs["model_name"] == "ViT":
            b_s = 128  # lowered batch size from 256 (original) to fit into memory
        else:
            b_s = 256

        blindspot_unlearner(
            model=student_model,
            unlearning_teacher=unlearning_teacher,
            full_trained_teacher=model,
            retain_data=retain_train_subset,
            forget_data=forget_train_dl.dataset,
            epochs=1,
            optimizer=optimizer,
            lr=0.0001,
            batch_size=b_s,
            device=device,
            KL_temperature=KL_temperature,
        )

        return get_metric_scores(
            student_model,
            unlearning_teacher,
            retain_train_dl,
            retain_valid_dl,
            forget_train_dl,
            forget_valid_dl,
            valid_dl,
            device,
        )

    # TODO: Adjust that
    def unlearn(self):
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
            acc_val = compute_accuracy(self.model, self.dl["val"], self.is_multi_label)

            # Log accuracies
            mlflow.log_metric("acc_retain", acc_retain, step=(epoch + 1))
            mlflow.log_metric("acc_val", acc_val, step=(epoch + 1))
            mlflow.log_metric("acc_forget", acc_forget, step=(epoch + 1))

        return self.model, run_time
