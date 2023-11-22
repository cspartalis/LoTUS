"""
This script performs the unlearning process of a using the original model.
It loads the original and the retrained model.
Early stopping when the accuracy on the forget set reaches the accuracy of the retrained model.
Epochs = epochs_to_retrain, warmup_epochs = 0.2 * epochs
It also computes the forgetting rate and the MIA metrics.
The script logs all the parameters and metrics to MLflow.
"""
import copy
import subprocess
import time

# pylint: disable=import-error
import warnings
from datetime import datetime

import mlflow
import torch
import torch.distributions as distributions
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from config import set_config
from data_utils import UnlearningDataLoader
from eval import (
    compute_accuracy,
    get_forgetting_rate,
    get_js_div,
    get_l2_params_distance,
    mia,
)
from mlflow_utils import mlflow_tracking_uri
from models import VGG19, AllCNN, ResNet18
from seed import set_seed

# pylint: enable=import-error
warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)
args = set_config()


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
            y[:, 0, :, :] = (y[:, 0, :, :] - self.mean[0]) / self.std[0]
            y[:, 1, :, :] = (y[:, 1, :, :] - self.mean[1]) / self.std[1]
            y[:, 2, :, :] = (y[:, 2, :, :] - self.mean[2]) / self.std[2]
            return y
        return x

    def inverse_normalize(self, x):
        if self.norm:
            y = x.clone().to(x.device)
            y[:, 0, :, :] = y[:, 0, :, :] * self.std[0] + self.mean[0]
            y[:, 1, :, :] = y[:, 1, :, :] * self.std[1] + self.mean[1]
            y[:, 2, :, :] = y[:, 2, :, :] * self.std[2] + self.mean[2]
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


if __name__ == "__main__":
    if args.run_id is None:
        raise ValueError("Please provide a run_id")

    # Start MLflow run
    now = datetime.now()
    str_now = now.strftime("%m-%d-%H-%M")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    retrain_run = mlflow.get_run(args.run_id)

    # Load params from retraining run
    seed = int(retrain_run.data.params["seed"])
    dataset = retrain_run.data.params["dataset"]
    model_str = retrain_run.data.params["model"]
    batch_size = int(retrain_run.data.params["batch_size"])
    epochs_to_retrain = int(retrain_run.data.metrics["best_epoch"])
    loss_str = retrain_run.data.params["loss"]
    optimizer_str = retrain_run.data.params["optimizer"]
    momentum = float(retrain_run.data.params["momentum"])
    weight_decay = float(retrain_run.data.params["weight_decay"])

    # Load params from config
    lr = args.lr

    set_seed(seed, args.cudnn)

    # Log parameters
    mlflow.set_experiment(f"{model_str}_{dataset}")
    mlflow.start_run(run_name=f"{model_str}_{dataset}_boundary_shrink_{str_now}")
    mlflow.log_param("reference_run_name", retrain_run.info.run_name)
    mlflow.log_param("reference_run_id", args.run_id)
    mlflow.log_param("seed", seed)
    mlflow.log_param("cudnn", args.cudnn)
    mlflow.log_param("dataset", dataset)
    mlflow.log_param("model", model_str)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("epochs", epochs_to_retrain)
    mlflow.log_param("loss", loss_str)
    mlflow.log_param("optimizer", optimizer_str)
    mlflow.log_param("lr", lr)
    mlflow.log_param("momentum", momentum)
    mlflow.log_param("weight_decay", weight_decay)

    commit_hash = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
    )
    mlflow.log_param("git_commit_hash", commit_hash)

    # Load data
    UDL = UnlearningDataLoader(dataset, batch_size, seed)
    dl, _ = UDL.load_data()
    num_classes = len(UDL.classes)
    input_channels = UDL.input_channels
    image_size = UDL.image_size

    # Load model architecture
    if model_str == "resnet18":
        model = ResNet18(input_channels, num_classes)
    elif model_str == "allcnn":
        model = AllCNN(input_channels, num_classes)
    elif model_str == "vgg19":
        model = VGG19(input_channels, num_classes)
    else:
        raise ValueError("Model not supported")
    # Load the original model
    model = mlflow.pytorch.load_model(f"{retrain_run.info.artifact_uri}/original_model")

    # Define loss function
    if loss_str == "cross_entropy":
        loss_fn = nn.CrossEntropyLoss()
    elif loss_str == "weighted_cross_entropy":
        samples_per_class = UDL.get_samples_per_class("retain")
        l_samples_per_class = list(samples_per_class.values())
        total_samples = sum(l_samples_per_class)
        # fmt: off
        class_weights = [total_samples / (num_classes * samples_per_class[i]) for i in range(num_classes)]
        class_weights = torch.FloatTensor(class_weights).to(DEVICE)
        # fmt: on
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # Set optimizer
    if optimizer_str == "sgd":
        optimizer = SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    elif optimizer_str == "adam":
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError("Optimizer not supported")

    # Set learning rate scheduler
    warmup_epochs = int(0.2 * epochs_to_retrain)
    mlflow.log_param("warmup_epochs", warmup_epochs)
    # fmt: off
    lr_lambda = lambda epoch: min(1.0, (epoch + 1) / args.warmup_epochs) * (1.0 - max(0.0, (epoch + 1) - args.warmup_epochs) / (args.epochs - args.warmup_epochs))
    # fmt: on
    lr_scheduler = LambdaLR(optimizer, lr_lambda)

    # ======================================================================================
    poison_epoch = 10
    extra_exp = None
    lambda_ = 0.7
    bias = -0.5
    slope = 5.0
    norm = True  # None#True if data_name != "mnist" else False
    random_start = False  # False if attack != "pgd" else True

    # start of my snippet
    acc_forget_retrain = int(retrain_run.data.metrics["acc_forget"])
    run_time = 0  # pylint: disable=invalid-name
    start_time = time.time()
    # end of my snippet

    test_model = copy.deepcopy(model).to(DEVICE)
    unlearn_model = copy.deepcopy(model).to(DEVICE)

    adv = FGSM(test_model, bound=0.1, norm=True, random_start=False, device=DEVICE)
    forget_data_gen = inf_generator(dl["forget"])
    batches_per_epoch = len(dl["forget"])

    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(unlearn_model.parameters(), lr=1e-5, momentum=0.9)  

    num_hits = 0
    num_sum = 0
    nearest_label = []
    best_epoch = None
    for epoch in tqdm(range(epochs_to_retrain)):
        for itr in range(batches_per_epoch):
            x, y = forget_data_gen.__next__()
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            test_model.eval()
            x_adv = adv.perturb(x, y, target_y=None, model=test_model, device=DEVICE)
            adv_logits = test_model(x_adv)
            pred_label = torch.argmax(adv_logits, dim=1)
            if itr >= batches_per_epoch - 1:
                nearest_label.append(pred_label.tolist())
            num_hits += (y != pred_label).float().sum()
            num_sum += y.shape[0]

            # adv_train
            unlearn_model.train()
            unlearn_model.zero_grad()
            optimizer.zero_grad()

            ori_logits = unlearn_model(x)
            loss = loss_fn(ori_logits, pred_label)

            loss.backward()
            optimizer.step()

        # start of my snippet
        epoch_run_time = (time.time() - start_time) / 60  # in minutes
        run_time += epoch_run_time

        acc_retain = compute_accuracy(model, dl["retain"])
        acc_forget = compute_accuracy(model, dl["forget"])
        acc_val = compute_accuracy(model, dl["val"])

        # Log accuracies
        mlflow.log_metric("acc_retain", acc_retain, step=epoch)
        mlflow.log_metric("acc_val", acc_val, step=epoch)
        mlflow.log_metric("acc_forget", acc_forget, step=epoch)

        if acc_forget <= acc_forget_retrain:
            best_epoch = epoch
            best_time = run_time
            break

        lr_scheduler.step()
        # end of my snippet
        
    # ======================================================================================

    # Save best model
    mlflow.pytorch.log_model(model, "unlearned_model")

    # Evaluation

    # Compute accuracy on the test dataset
    acc_test = compute_accuracy(model, dl["test"])

    # Load the retrained model (is needed for js_div, l2_params_distance, and mia)
    retrained_model = mlflow.pytorch.load_model(
        f"{retrain_run.info.artifact_uri}/retrained_model"
    )

    # Compute the js_div, l2_params_distance
    js_div = get_js_div(retrained_model, model, dl["forget"])
    l2_params_distance = get_l2_params_distance(retrained_model, model)

    # Load tp and fn of the original model
    original_tp = int(retrain_run.data.params["original_tp"])
    original_fn = int(retrain_run.data.params["original_fn"])
    # Load the training loss of the original model to be used as threshold for mia
    original_tr_loss_threshold = float(
        retrain_run.data.params["original_tr_loss_threshold"]
    )

    # Compute the MIA metrics and Forgetting rate
    mia_bacc, mia_tpr, mia_fpr, mia_tp, mia_fn = mia(
        model, dl["forget"], dl["val"], original_tr_loss_threshold, num_classes
    )
    forgetting_rate = get_forgetting_rate(original_tp, original_fn, mia_fn)

    # Log metrics
    if best_epoch is None:
        best_epoch = epochs_to_retrain
        best_time = run_time
    mlflow.log_metric("best_epoch", best_epoch)
    mlflow.log_metric("best_time", round(best_time, 2))
    mlflow.log_metric("acc_test", acc_test)
    mlflow.log_metric("js_div", js_div)
    mlflow.log_metric("l2_params_distance", l2_params_distance)
    mlflow.log_metric("mia_balanced_acc", mia_bacc)
    mlflow.log_metric("mia_tpr", mia_tpr)
    mlflow.log_metric("mia_fpr", mia_fpr)
    mlflow.log_metric("mia_tp", mia_tp)
    mlflow.log_metric("mia_fn", mia_fn)
    mlflow.log_metric("forgetting_rate", forgetting_rate)

    mlflow.end_run()
