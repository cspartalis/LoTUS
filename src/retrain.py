"""
Implentation of retraining without the forget set.
"""

import subprocess
import time

# pylint: disable=import-error
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
from eval import (
    compute_accuracy,
    get_forgetting_rate,
    get_js_div,
    get_l2_params_distance,
    mia,
)
from mlflow_utils import mlflow_tracking_uri
from seed import set_seed

# pylint: enable=import-error

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)
args = set_config()

if args.run_id is None:
    raise ValueError("Please provide a run_id")

# Start MLflow run
now = datetime.now()
str_now = now.strftime("%m-%d-%H-%M")
mlflow.set_tracking_uri(mlflow_tracking_uri)

original_run = mlflow.get_run(args.run_id)
model_str = original_run.data.params["model"]
dataset = original_run.data.params["dataset"]
mlflow.set_experiment(f"{model_str}_{dataset}")

mlflow.start_run(run_name="retrained")

# Load params from original run
seed = int(original_run.data.params["seed"])
batch_size = int(original_run.data.params["batch_size"])
epochs = int(original_run.data.params["epochs"])
loss_str = original_run.data.params["loss"]
lr = float(original_run.data.params["lr"])
optimizer_str = original_run.data.params["optimizer"]
momentum = float(original_run.data.params["momentum"])
weight_decay = float(original_run.data.params["weight_decay"])
is_lr_scheduler = original_run.data.params["is_lr_scheduler"]
is_lr_scheduler = is_lr_scheduler.lower() == "true"
if is_lr_scheduler:
    warmup_epochs = int(original_run.data.params["warmup_epochs"])
is_early_stop = original_run.data.params["is_early_stop"]
is_early_stop = is_early_stop.lower() == "true"
if is_early_stop:
    patience = int(original_run.data.params["patience"])

set_seed(seed, args.cudnn)

# Log parameters
mlflow.log_param("datetime", str_now)
mlflow.log_param("reference_run_name", original_run.info.run_name)
mlflow.log_param("reference_run_id", args.run_id)
mlflow.log_param("seed", seed)
mlflow.log_param("cudnn", args.cudnn)
mlflow.log_param("dataset", dataset)
mlflow.log_param("model", model_str)
mlflow.log_param("batch_size", batch_size)
mlflow.log_param("epochs", epochs)
mlflow.log_param("loss", loss_str)
mlflow.log_param("optimizer", optimizer_str)
mlflow.log_param("lr", lr)
mlflow.log_param("momentum", momentum)
mlflow.log_param("weight_decay", weight_decay)
mlflow.log_param("is_lr_scheduler", is_lr_scheduler)
if is_lr_scheduler:
    mlflow.log_param("warmup_epochs", warmup_epochs)
mlflow.log_param("is_early_stop", is_early_stop)
if is_early_stop:
    mlflow.log_param("patience", patience)
commit_hash = (
    subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
)
mlflow.log_param("git_commit_hash", commit_hash)

# Load model and data
if model_str == "resnet18":
    if dataset == "cifar-10" or dataset == "cifar-100":
        image_size = 32
    elif dataset == "mufac":
        image_size = 128
    elif dataset == "pneumoniamnist":
        image_size = 224
    else:
        raise ValueError("Dataset not supported")

    UDL = UnlearningDataLoader(dataset, batch_size, image_size, seed, is_vit=False)
    dl, _ = UDL.load_data()
    num_classes = len(UDL.classes)
    input_channels = UDL.input_channels

    from models import ResNet18

    model = ResNet18(input_channels, num_classes)

elif model_str == "vit":
    image_size = 224

    UDL = UnlearningDataLoader(dataset, batch_size, image_size, seed, is_vit=True)
    dl, _ = UDL.load_data()
    num_classes = len(UDL.classes)

    from models import ViT

    model = ViT(num_classes=num_classes)
else:
    raise ValueError("Model not supported")

if dataset == "mucac":
    # Define a custom head for the multi-label classification.

    class MultiLabelHead(nn.Module):
        def __init__(self, in_features, out_features):
            super(MultiLabelHead, self).__init__()
            self.fc = nn.Linear(in_features, out_features)

        def forward(self, x):
            return self.fc(x)

    if model_str == "resnet18":
        num_features = model.fc.in_features
        model.fc = MultiLabelHead(num_features, num_classes)
    elif model_str == "vit":
        num_features = model.final.in_features
        model.final = MultiLabelHead(num_features, num_classes)


# Define loss function
if loss_str == "cross_entropy":
    loss_fn = nn.CrossEntropyLoss()
elif loss_str == "weighted_cross_entropy":
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
if optimizer_str == "sgd":
    optimizer = SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
elif optimizer_str == "adam":
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
else:
    raise ValueError("Optimizer not supported")

# Linear decay learning rate scheduler with warmup
if is_lr_scheduler:
    # fmt: off
    lr_lambda = lambda epoch: min(1.0, (epoch + 1) / warmup_epochs) * (1.0 - max(0.0, (epoch + 1) - warmup_epochs) / (args.epochs - warmup_epochs))  # pylint: disable=line-too-long
    # fmt: on
    lr_scheduler = LambdaLR(optimizer, lr_lambda)

# Train on retain set
model.to(DEVICE)
best_val_loss = float("inf")
run_time = 0
for epoch in tqdm(range(epochs)):
    start_time = time.time()
    model.train()
    train_loss = 0.0  # pylint: disable=invalid-name
    for inputs, targets in dl["retain"]:
        inputs = inputs.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(dl["retain"])
    epoch_run_time = (time.time() - start_time) / 60  # in minutes
    run_time += epoch_run_time

    model.eval()
    with torch.inference_mode():
        val_loss = 0  # pylint: disable=invalid-name
        for inputs, targets in dl["val"]:
            inputs = inputs.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            val_loss += loss.item()
        val_loss /= len(dl["val"])

    # print(
    #     f"Epoch: {epoch + 1} | Train Loss: {train_loss:.3f} | Val loss: {val_loss:.3f}"
    # )

    # Log losses
    mlflow.log_metric("train_loss", train_loss, step=epoch)
    mlflow.log_metric("val_loss", val_loss, step=epoch)

    if is_early_stop:
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            best_epoch = epoch
            best_time = run_time
            epochs_no_improve = 0  # pylint: disable=invalid-name
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                break

    if is_lr_scheduler:
        lr_scheduler.step()

# Save best model
if is_early_stop:
    model.load_state_dict(best_model)
mlflow.pytorch.log_model(model, "retrained_model")

# Evaluation
# Load the original model (is needed for js_div, l2_params_distance, and mia)
original_model = mlflow.pytorch.load_model(
    f"{original_run.info.artifact_uri}/original_model"
)

# Log the original model to be used in the following experiments
mlflow.pytorch.log_model(original_model, "original_model")

# Load tp and fn of the original model
original_tp = int(original_run.data.metrics["mia_tp"])
original_fn = int(original_run.data.metrics["mia_fn"])
# Load the training loss of the original model to be used as threshold for mia
original_tr_loss_threshold = float(
    original_run.data.metrics["original_tr_loss_threshold"]
)

# Compute the accuracy metrics
is_multi_label = True if dataset == "mucac" else False
acc_retain = compute_accuracy(model, dl["retain"], is_multi_label)
acc_val = compute_accuracy(model, dl["val"], is_multi_label)
acc_forget = compute_accuracy(model, dl["forget"], is_multi_label)
acc_test = compute_accuracy(model, dl["test"], is_multi_label)

# Compute the js_div, l2_params_distance
js_div = get_js_div(original_model, model, dl["forget"])
l2_params_distance, l2_params_distance_norm = get_l2_params_distance(
    model, original_model
)

# Compute the MIA metrics and Forgetting rate
mia_bacc, mia_tpr, mia_tnr, mia_tp, mia_fn = mia(
    model, dl["forget"], dl["val"], original_tr_loss_threshold
)
forgetting_rate = get_forgetting_rate(original_tp, original_fn, mia_fn)

# Log metrics
mlflow.log_metric("best_epoch", best_epoch)
mlflow.log_metric("best_time", round(best_time, 2))
mlflow.log_metric("acc_retain", acc_retain)
mlflow.log_metric("acc_val", acc_val)
mlflow.log_metric("acc_forget", acc_forget)
mlflow.log_metric("acc_test", acc_test)
mlflow.log_metric("js_div", js_div)
mlflow.log_metric("l2_params_distance", l2_params_distance)
mlflow.log_metric("l2_params_distance_norm", l2_params_distance_norm)
mlflow.log_metric("mia_acc", mia_bacc)
mlflow.log_metric("mia_tpr", mia_tpr)
mlflow.log_metric("mia_tnr", mia_tnr)
mlflow.log_metric("mia_tp", mia_tp)
mlflow.log_metric("mia_fn", mia_fn)
mlflow.log_param("original_tp", original_tp)
mlflow.log_param("original_fn", original_fn)
mlflow.log_param("original_tr_loss_threshold", original_tr_loss_threshold)
mlflow.log_metric("forgetting_rate", forgetting_rate)

mlflow.end_run()
