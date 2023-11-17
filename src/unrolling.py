"""
This script performs the unlearning process of a using the original model.
It loads the original and the retrained model, and fine-tunes the original model on the retain set.
Early stopping when the accuracy on the forget set reaches the accuracy of the retrained model.
Epochs = epochs_to_retrain, warmup_epochs = 0.2 * epochs
It also computes the forgetting rate and the MIA metrics.
The script logs all the parameters and metrics to MLflow.
"""
import subprocess
import time

# pylint: disable=import-error
import warnings
from datetime import datetime

import mlflow
import torch
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
    get_l2_weight_distance,
    mia,
)
from mlflow_utils import mlflow_tracking_uri
from models import VGG19, AllCNN, ResNet18
from seed import set_seed

# pylint: enable=import-error

def std_loss(x, y, std_reg=5):
    """
    Computes the standard deviation loss for a given input and target.

    Args:
        x (torch.Tensor): Input tensor.
        y (torch.Tensor): Target tensor.

    Returns:
        torch.Tensor: Computed loss tensor.
    """
    log_prob = -1.0 * F.log_softmax(x, 1)
    loss = log_prob.gather(1, y.unsqueeze(1))
    loss = loss.mean()
    avg_std = torch.sum(torch.std(x, dim=1)) / (len(x.view(-1)))
    loss = loss + std_reg * avg_std
    return loss


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
retrain_run = mlflow.get_run(args.run_id)

# Load params from retraining run
seed = int(retrain_run.data.params["seed"])
dataset = retrain_run.data.params["dataset"]
model_str = retrain_run.data.params["model"]
batch_size = int(retrain_run.data.params["batch_size"])
epochs_to_retrain = int(retrain_run.data.metrics["best_epoch"])
# loss_str = retrain_run.data.params["loss"]
optimizer_str = retrain_run.data.params["optimizer"]
momentum = float(retrain_run.data.params["momentum"])
weight_decay = float(retrain_run.data.params["weight_decay"])

# Load params from config
lr = args.lr

set_seed(seed, args.cudnn)

# Log parameters
mlflow.set_experiment(f"{model_str}_{dataset}")
mlflow.start_run(run_name=f"{model_str}_{dataset}_unrolling_{str_now}")
mlflow.log_param("reference_run_name", retrain_run.info.run_name)
mlflow.log_param("reference_run_id", args.run_id)
mlflow.log_param("seed", seed)
mlflow.log_param("cudnn", args.cudnn)
mlflow.log_param("dataset", dataset)
mlflow.log_param("model", model_str)
mlflow.log_param("batch_size", batch_size)
mlflow.log_param("epochs", epochs_to_retrain)
mlflow.log_param("loss", 'std_loss')
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


# Train on retain set
model.to(DEVICE)
acc_forget_retrain = int(retrain_run.data.metrics["acc_forget"])
run_time = 0  # pylint: disable=invalid-name
for epoch in tqdm(range(epochs_to_retrain)):
    start_time = time.time()
    model.train()
    for _, (inputs, targets) in enumerate(dl["retain"]):
        inputs = inputs.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = std_loss(outputs, targets)
        loss.backward()
        optimizer.step()
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

# Save best model
mlflow.pytorch.log_model(model, "unlearned_model")

# Evaluation

# Compute accuracy on the test dataset
acc_test = compute_accuracy(model, dl["test"])

# Load the retrained model (is needed for js_div, l2_weight_distance, and mia)
retrained_model = mlflow.pytorch.load_model(
    f"{retrain_run.info.artifact_uri}/retrained_model"
)

# Compute the js_div, l2_weight_distance
js_div = get_js_div(retrained_model, model, dl["forget"])
l2_weight_distance = get_l2_weight_distance(retrained_model, model)

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
forgetting_rate = get_forgetting_rate(bt=original_tp, bf=original_fn, af=mia_fn)

# Log metrics
mlflow.log_metric("best_epoch", best_epoch)
mlflow.log_metric("best_time", round(best_time, 2))
mlflow.log_metric("acc_test", acc_test)
mlflow.log_metric("js_div", js_div)
mlflow.log_metric("l2_weight_distance", l2_weight_distance)
mlflow.log_metric("mia_balanced_acc", mia_bacc)
mlflow.log_metric("mia_tpr", mia_tpr)
mlflow.log_metric("mia_fpr", mia_fpr)
mlflow.log_metric("mia_tp", mia_tp)
mlflow.log_metric("mia_fn", mia_fn)
mlflow.log_metric("forgetting_rate", forgetting_rate)
