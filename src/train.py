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

from config import set_config
from data_utils import UnlearningDataLoader
from eval import compute_accuracy, mia
from mlflow_utils import mlflow_tracking_uri
from seed import set_seed

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
mlflow.set_experiment(f"{args.model}_{args.dataset}")
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

# Load model and data
if args.model == "resnet18":
    if args.dataset == "cifar-10" or args.dataset == "cifar-100":
        image_size = 32
    elif args.dataset == "mufac" or args.dataset == "mucac":
        image_size = 128
    elif args.dataset == "mnist":
        image_size = 28
    elif args.dataset == "pneumoniamnist":
        image_size = 224
    else:
        raise ValueError("Dataset not supported")

    UDL = UnlearningDataLoader(args.dataset, args.batch_size, image_size, args.seed)
    dl, _ = UDL.load_data()
    num_classes = len(UDL.classes)
    input_channels = UDL.input_channels

    from models import ResNet18

    model = ResNet18(input_channels, num_classes)

elif args.model == "vit":
    image_size = 224

    UDL = UnlearningDataLoader(args.dataset, args.batch_size, image_size, args.seed)
    dl, _ = UDL.load_data()
    num_classes = len(UDL.classes)

    from models import ViT

    model = ViT(num_classes=num_classes)
else:
    raise ValueError("Model not supported")

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
    lr_lambda = lambda epoch: min(1.0, (epoch + 1) / args.warmup_epochs) * (1.0 - max(0.0, (epoch + 1) - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
    )
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

    print(
        f"Epoch: {epoch + 1} | Train Loss: {train_loss:.3f} | Val loss: {val_loss:.3f}"
    )

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
mlflow.pytorch.log_model(model, "original_model")


# Evaluation
# Epochs and time
mlflow.log_metric("best_epoch", best_epoch)
best_time = round(best_time, 2)
mlflow.log_metric("best_time", best_time)

# Accuracies
acc_retain = compute_accuracy(model, dl["retain"])
acc_forget = compute_accuracy(model, dl["forget"])
acc_test = compute_accuracy(model, dl["test"])
acc_val = compute_accuracy(model, dl["val"])

acc_retain = round(acc_retain, 2)
acc_forget = round(acc_forget, 2)
acc_test = round(acc_test, 2)
acc_val = round(acc_val, 2)

mlflow.log_metric("acc_retain", acc_retain)
mlflow.log_metric("acc_forget", acc_forget)
mlflow.log_metric("acc_test", acc_test)
mlflow.log_metric("acc_val", acc_val)

# MIA
# Get the last train loss of the best_model
original_tr_loss = 0  # pylint: disable=invalid-name
model.eval()
with torch.inference_mode():
    for inputs, targets in dl["train"]:
        inputs = inputs.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        original_tr_loss += loss.item()
    original_tr_loss /= len(dl["train"])

mlflow.log_metric("original_tr_loss_threshold", original_tr_loss)

mia_acc, mia_tpr, mia_tnr, mia_tp, mia_fn = mia(
    model, dl["forget"], dl["val"], threshold=original_tr_loss
)

mlflow.log_metric("mia_acc", mia_acc)
mlflow.log_metric("mia_tpr", mia_tpr)
mlflow.log_metric("mia_tnr", mia_tnr)
mlflow.log_metric("mia_tp", mia_tp)
mlflow.log_metric("mia_fn", mia_fn)


# End MLflow run
mlflow.end_run()
