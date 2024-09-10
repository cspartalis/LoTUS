"""
Implentation of retraining without the forget set.
"""

import logging
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
from eval import compute_accuracy, log_l2_params_distance, log_membership_attack_prob
from mlflow_utils import mlflow_tracking_uri
from seed import set_seed

# log = logging.getLogger(__name__)
# logging.basicConfig(
#     filename="_debug.log",
#     filemode="w",
#     level=logging.INFO,
#     datefmt="%H:%M",
#     format="%(name)s - %(levelname)s - %(message)s",
# )

# pylint: enable=import-error

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)
args = set_config()

# Start MLflow run
now = datetime.now()
str_now = now.strftime("%m-%d-%H-%M")
mlflow.set_tracking_uri(mlflow_tracking_uri)

registered_model = args.registered_model
version = "latest"
try:
    model = mlflow.pytorch.load_model(model_uri=f"models:/{registered_model}/{version}")
except:
    raise ValueError("Model not found")

_ = mlflow.pyfunc.load_model(model_uri=f"models:/{registered_model}/{version}")
original_run_id = _.metadata.run_id

original_run = mlflow.get_run(original_run_id)
model_str = original_run.data.params["model"]
dataset = original_run.data.params["dataset"]
seed = int(original_run.data.params["seed"])
is_class_unlearning = args.is_class_unlearning
class_to_forget = args.class_to_forget
if is_class_unlearning:
    mlflow.set_experiment(f"_{model_str}_{class_to_forget}_{seed}")
else:
    mlflow.set_experiment(f"_{model_str}_{dataset}_{seed}")

mlflow.start_run(run_name="retrained")

# Load params from original run
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
mlflow.log_param("reference_run_id", original_run_id)
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
mlflow.log_param("is_class_unlearning", is_class_unlearning)
mlflow.log_param("class_to_forget", class_to_forget)
mlflow.log_param("method", "retrained")

# Load model and data
if model_str == "resnet18":
    if dataset in ["cifar-10", "cifar-100"]:
        image_size = 32
    elif dataset in ["mufac", "mucac", "pneumoniamnist"]:
        image_size = 128
    else:
        raise ValueError("Dataset not supported")

    UDL = UnlearningDataLoader(
        dataset,
        batch_size,
        image_size,
        seed,
        is_vit=False,
        is_class_unlearning=is_class_unlearning,
        class_to_forget=class_to_forget,
    )
    dl, _ = UDL.load_data()
    num_classes = len(UDL.classes)
    input_channels = UDL.input_channels

    from models import ResNet18

    model = ResNet18(input_channels, num_classes)

elif model_str == "vit":
    image_size = 224

    UDL = UnlearningDataLoader(
        dataset,
        batch_size,
        image_size,
        seed,
        is_vit=True,
        is_class_unlearning=is_class_unlearning,
        class_to_forget=class_to_forget,
    )
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
elif loss_str == "bce_with_logits":
    loss_fn = nn.BCEWithLogitsLoss()

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
    lr_lambda = lambda epoch: min(1.0, (epoch + 1) / warmup_epochs) * (1.0 - max(0.0, (epoch + 1) - warmup_epochs) / (epochs - warmup_epochs))  # pylint: disable=line-too-long
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


if is_class_unlearning:
    mlflow.pytorch.log_model(
        model,
        artifact_path="models",
        registered_model_name=f"{model_str}-{dataset}-{class_to_forget}-{seed}-retrained",
    )
else:
    mlflow.pytorch.log_model(
        model,
        artifact_path="models",
        registered_model_name=f"{model_str}-{dataset}-{seed}-retrained",
    )

# Log original
original_model = mlflow.pytorch.load_model(
    f"{original_run.info.artifact_uri}/original_model"
)
mlflow.pytorch.log_model(original_model, "original_model")

# Evaluation

# Compute the accuracy metrics
is_multi_label = True if dataset == "mucac" else False
acc_retain = compute_accuracy(model, dl["retain"], is_multi_label)
acc_val = compute_accuracy(model, dl["val"], is_multi_label)
acc_forget = compute_accuracy(model, dl["forget"], is_multi_label)
acc_test = compute_accuracy(model, dl["test"], is_multi_label)


# Log metrics
mlflow.log_metric("best_epoch", best_epoch)
mlflow.log_metric("t", round(best_time, 2))
mlflow.log_metric("acc_retain", acc_retain)
mlflow.log_metric("acc_val", acc_val)
mlflow.log_metric("acc_forget", acc_forget)
mlflow.log_metric("acc_test", acc_test)
# Check streisand effect (L2 distances between original and unlearned model)
l2 = log_l2_params_distance(model, original_model)
mlflow.log_metric("l2", l2)

log_membership_attack_prob(dl["retain"], dl["forget"], dl["test"], dl["val"], model)
# Log MIA and accuracies for original model
if is_class_unlearning:
    log_membership_attack_prob(
        dl["retain"],
        dl["forget"],
        dl["test"],
        dl["val"],
        original_model,
        step=None,
        is_original=True,
    )
    original_forget_acc = compute_accuracy(original_model, dl["forget"])
    original_retain_acc = compute_accuracy(original_model, dl["forget"])

    mlflow.log_metric("original_forget_acc", original_forget_acc)
    mlflow.log_metric("original_retain_acc", original_retain_acc)

mlflow.end_run()
