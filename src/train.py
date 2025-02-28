"""
This script trains a deep learning model on a given dataset using PyTorch.
The model can be one of the following: ResNet18, AllCNN, or LiteViT.
The dataset can be one of the following: CIFAR10, CIFAR100, or Tiny-ImageNet.
The training can be done from scratch or by retraining without the forget set.
The script logs the training process and the evaluation metrics using MLflow.
The best model is saved as a PyTorch model and checkpoints are saved during training.
Early stopping can be enabled to stop training when the validation loss does not improve for a given number of epochs.
"""

import subprocess

# pylint: disable=import-error
import time
import warnings
from datetime import datetime

import mlflow
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from helpers.config import set_config
from helpers.data_utils import UnlearningDataLoader
from helpers.eval import compute_accuracy, log_js_proxy, log_mia
from helpers.mlflow_utils import mlflow_tracking_uri
from helpers.models import ResNet18, ViT
from helpers.seed import set_seed

# pylint: disable=enable-error


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)
args = set_config()
set_seed(args.seed, args.cudnn)
warnings.filterwarnings("ignore")

# Start MLflow run
now = datetime.now()
str_now = now.strftime("%m-%d-%H-%M")
mlflow.set_tracking_uri(mlflow_tracking_uri)
if args.is_class_unlearning:
    mlflow.set_experiment(
        f"cs_{args.model}_{args.dataset}_{args.class_to_forget}"
    )
else:
    mlflow.set_experiment(f"cs_{args.model}_{args.dataset}")
mlflow.start_run(run_name="original")


# Log parameters
mlflow.log_param("datetime", str_now)
mlflow.log_param("seed", args.seed)
mlflow.log_param("cudnn", args.cudnn)
mlflow.log_param("dataset", args.dataset)
mlflow.log_param("model", args.model)
mlflow.log_param("batch_size", args.batch_size)
mlflow.log_param("epochs", args.epochs)
mlflow.log_param("loss", args.loss)
mlflow.log_param("optimizer", args.optimizer)
mlflow.log_param("lr", args.lr)
mlflow.log_param("momentum", args.momentum)
mlflow.log_param("weight_decay", args.weight_decay)
mlflow.log_param("is_lr_scheduler", args.is_lr_scheduler)
if args.is_lr_scheduler:
    mlflow.log_param("warmup_epochs", args.warmup_epochs)
mlflow.log_param("is_early_stop", args.is_early_stop)
if args.is_early_stop:
    mlflow.log_param("patience", args.patience)
commit_hash = (
    subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
)
mlflow.log_param("git_commit_hash", commit_hash)
mlflow.log_param("is_class_unlearning", args.is_class_unlearning)
mlflow.log_param("class_to_forget", args.class_to_forget)
mlflow.log_param("method", "original")

# Load model and data
if args.model == "resnet18":
    if args.dataset in ["cifar-10", "cifar-100"]:
        image_size = 32
    elif args.dataset == "mufac":
        image_size = 128
    elif args.dataset == "tiny-imagenet":
        image_size = 64
    else:
        raise ValueError("Dataset not supported")

    UDL = UnlearningDataLoader(
        args.dataset,
        args.batch_size,
        image_size,
        args.seed,
        is_vit=False,
        is_class_unlearning=args.is_class_unlearning,
        class_to_forget=args.class_to_forget,
    )
    dl, _ = UDL.load_data()
    num_classes = len(UDL.classes)
    input_channels = UDL.input_channels
    if isinstance(input_channels, tuple):
        input_channels = input_channels[0]

    model = ResNet18(input_channels, num_classes)

elif args.model == "vit":
    image_size = 224

    UDL = UnlearningDataLoader(
        args.dataset,
        args.batch_size,
        image_size,
        args.seed,
        is_vit=True,
        is_class_unlearning=args.is_class_unlearning,
        class_to_forget=args.class_to_forget,
    )
    dl, _ = UDL.load_data()
    num_classes = len(UDL.classes)

    model = ViT(num_classes=num_classes)
else:
    raise ValueError("Model not supported")

if args.dataset == "mucac":
    # Define a custom head for the multi-label classification.

    class MultiLabelHead(nn.Module):
        def __init__(self, in_features, out_features):
            super(MultiLabelHead, self).__init__()
            self.fc = nn.Linear(in_features, out_features)

        def forward(self, x):
            return self.fc(x)

    if args.model == "resnet18":
        num_features = model.fc.in_features
        model.fc = MultiLabelHead(num_features, num_classes)
    elif args.model == "vit":
        num_features = model.final.in_features
        model.final = MultiLabelHead(num_features, num_classes)


# Define loss function
if args.loss == "cross_entropy":
    loss_fn = nn.CrossEntropyLoss()
elif args.loss == "weighted_cross_entropy":
    samples_per_class = UDL.get_samples_per_class("train")
    l_samples_per_class = list(samples_per_class.values())
    total_samples = sum(l_samples_per_class)
    class_weights = [
        total_samples / (num_classes * samples_per_class[i]) for i in range(num_classes)
    ]
    class_weights = torch.FloatTensor(class_weights)
    class_weights = class_weights.to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
elif args.loss == "bce_with_logits":
    loss_fn = nn.BCEWithLogitsLoss()

# Set optimizer and learning rate scheduler
if args.optimizer == "sgd":
    optimizer = SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
elif args.optimizer == "adam":
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
else:
    raise ValueError("Optimizer not supported")

# Linear decay learning rate scheduler with warmup
if args.is_lr_scheduler:
    # fmt: off
    lr_lambda = lambda epoch: min(1.0, (epoch + 1) / args.warmup_epochs) * (1.0 - max(0.0, (epoch + 1) - args.warmup_epochs) / (args.epochs - args.warmup_epochs))
    # fmt: on
    lr_scheduler = LambdaLR(optimizer, lr_lambda)


# Train model
model.to(DEVICE)
best_val_loss = float("inf")
run_time = 0  # in minutes
for epoch in tqdm(range(args.epochs)):
    start_time = time.time()
    model.train()
    train_loss = 0  # pylint: disable=invalid-name
    for inputs, targets in dl["train"]:
        inputs = inputs.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(dl["train"])
    epoch_run_time = (time.time() - start_time) / 60  # in minutes
    run_time += epoch_run_time

    model.eval()
    val_loss = 0
    with torch.inference_mode():
        for inputs, targets in dl["val"]:
            inputs = inputs.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            val_loss += loss.item()
        val_loss /= len(dl["val"])

    # Log losses
    mlflow.log_metric("train_loss", train_loss, step=epoch)
    mlflow.log_metric("val_loss", val_loss, step=epoch)

    if args.is_early_stop:
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            best_epoch = epoch
            best_time = run_time
            epochs_no_improve = 0  # pylint: disable=invalid-name
        else:
            epochs_no_improve += 1
            if epochs_no_improve == args.patience:
                break

    if args.is_lr_scheduler:
        lr_scheduler.step()

# Save best model
if args.is_early_stop:
    model.load_state_dict(best_model)
    mlflow.log_metric("best_epoch", best_epoch)
mlflow.pytorch.log_model(model, "original_model")
mlflow.pytorch.log_model(
    model,
    artifact_path="models",
    registered_model_name=f"{args.model}-{args.dataset}-{args.seed}-original",
)

# Evaluation
mia = log_mia(dl["retain"], dl["forget"], dl["test"], dl["val"], model)
js = log_js_proxy(model, model, dl["forget"], dl["test"])

time = round(best_time, 2) if args.is_early_stop else round(run_time, 2)
mlflow.log_metric("t", time)

acc_retain = compute_accuracy(model, dl["retain"])
mlflow.log_metric("acc_retain", acc_retain)

acc_forget = compute_accuracy(model, dl["forget"])
mlflow.log_metric("acc_forget", acc_forget)

acc_test = compute_accuracy(model, dl["test"])
mlflow.log_metric("acc_test", acc_test)

acc_val = compute_accuracy(model, dl["val"])
mlflow.log_metric("acc_val", acc_val)


# End MLflow run
mlflow.end_run()
