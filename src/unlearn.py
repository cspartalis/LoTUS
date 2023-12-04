"""
This script performs the unlearning process 
"""
# pylint: disable=import-error
import subprocess
import time
import warnings
from datetime import datetime

import mlflow
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR

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
from models import VGG19, AllCNN, ResNet18, ViT
from seed import set_seed
from unlearning_class import UnlearningClass

# pylint: enable=import-error

# ==== SETUP ====

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
loss_str = retrain_run.data.params["loss"]
optimizer_str = retrain_run.data.params["optimizer"]
momentum = float(retrain_run.data.params["momentum"])
weight_decay = float(retrain_run.data.params["weight_decay"])
acc_forget_retrain = int(retrain_run.data.metrics["acc_forget"])

# Load params from config
lr = args.lr
epochs = args.epochs
set_seed(seed, args.cudnn)

# Log parameters
mlflow.set_experiment(f"{model_str}_{dataset}")
mlflow.start_run(run_name=f"{model_str}_{dataset}_{args.mu_method}_{str_now}")
mlflow.log_param("reference_run_name", retrain_run.info.run_name)
mlflow.log_param("reference_run_id", args.run_id)
mlflow.log_param("seed", seed)
mlflow.log_param("cudnn", args.cudnn)
mlflow.log_param("dataset", dataset)
mlflow.log_param("model", model_str)
mlflow.log_param("batch_size", batch_size)
mlflow.log_param("epochs", args.epochs)
mlflow.log_param("loss", loss_str)
mlflow.log_param("optimizer", optimizer_str)
mlflow.log_param("lr", lr)
mlflow.log_param("momentum", momentum)
mlflow.log_param("weight_decay", weight_decay)
mlflow.log_param("is_lr_scheduler", args.is_lr_scheduler)
mlflow.log_param("is_early_stop", args.is_early_stop)
mlflow.log_param("mu_method", args.mu_method)

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
elif model_str == "vit":
    model = ViT(image_size=image_size, num_classes=num_classes)
else:
    raise ValueError("Model not supported")
# Load the original model
model = mlflow.pytorch.load_model(f"{retrain_run.info.artifact_uri}/original_model")
model.to(DEVICE)

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
else:
    raise ValueError("Loss function not supported")

# Set optimizer
if optimizer_str == "sgd":
    optimizer = SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
elif optimizer_str == "adam":
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
else:
    raise ValueError("Optimizer not supported")

# Set learning rate scheduler (if any)
if args.is_lr_scheduler:
    warmup_epochs = int(0.2 * epochs_to_retrain)
    mlflow.log_param("warmup_epochs", warmup_epochs)
    # fmt: off
    lr_lambda = lambda epoch: min(1.0, (epoch + 1) / args.warmup_epochs) * (1.0 - max(0.0, (epoch + 1) - args.warmup_epochs) / (args.epochs - args.warmup_epochs))
    # fmt: on
    lr_scheduler = LambdaLR(optimizer, lr_lambda)
else:
    lr_scheduler = None

# ==== UNLEARNING ====

if args.mu_method == "zapping":
    dl_start_prep_time = time.time()
    dl["mock_forget"] = UDL.get_mock_forget_dataloader(model)
    dl_prep_time = (time.time() - dl_start_prep_time) / 60  # in minutes
elif args.mu_method == "relabel":
    dl_start_prep_time = time.time()
    dl["mixed"] = UDL.get_mixed_dataloader(model)
    dl_prep_time = (time.time() - dl_start_prep_time) / 60  # in minutes

UC = UnlearningClass(
    dl,
    batch_size,
    num_classes,
    model,
    epochs,
    loss_fn,
    optimizer,
    lr_scheduler,
    acc_forget_retrain,
    args.is_early_stop,
)

match args.mu_method:
    case "finetune":
        model, epoch, run_time = UC.finetune()
    case "neggrad":
        model, epoch, run_time = UC.neggrad()
    case "relabel":
        model, epoch, run_time = UC.relabel(dl_prep_time)
    case "boundary":
        model, epoch, run_time = UC.boundary()
    case "zapping":
        mlflow.log_param("zap_is_diff_grads", args.is_diff_grads)
        mlflow.log_param("zap_threshold", args.zap_threshold)
        model, epoch, run_time = UC.zapping(
            args.is_diff_grads, args.zap_threshold, dl_prep_time
        )

# Save the unlearned model
mlflow.pytorch.log_model(model, "unlearned_model")

# ==== EVALUATION =====

# Compute accuracy on the test dataset
acc_test = compute_accuracy(model, dl["test"])

# Load the retrained model (is needed for js_div, l2_params_distance, and mia)
retrained_model = mlflow.pytorch.load_model(
    f"{retrain_run.info.artifact_uri}/retrained_model"
)

# Compute the js_div, l2_params_distance
js_div = get_js_div(retrained_model, model, dl["forget"])
l2_params_distance, l2_params_distance_norm = get_l2_params_distance(
    retrained_model, model
)

# Load tp and fn of the original model
original_tp = int(retrain_run.data.params["original_tp"])
original_fn = int(retrain_run.data.params["original_fn"])
# Load the training loss of the original model to be used as threshold for mia
original_tr_loss_threshold = float(
    retrain_run.data.params["original_tr_loss_threshold"]
)

# Compute the MIA metrics and Forgetting rate
mia_bacc, mia_tpr, mia_tnr, mia_tp, mia_fn = mia(
    model, dl["forget"], dl["val"], original_tr_loss_threshold, num_classes
)
forgetting_rate = get_forgetting_rate(original_tp, original_fn, mia_fn)

# Log metrics
mlflow.log_metric("epoch_unlearn", epoch)
mlflow.log_metric("time_unlearn", round(run_time, 2))
mlflow.log_metric("acc_test", acc_test)
mlflow.log_metric("js_div", js_div)
mlflow.log_metric("l2_params_distance", l2_params_distance)
mlflow.log_metric("l2_params_distance_norm", l2_params_distance_norm)
mlflow.log_metric("mia_acc", mia_bacc)
mlflow.log_metric("mia_tpr", mia_tpr)
mlflow.log_metric("mia_tnr", mia_tnr)
mlflow.log_metric("mia_tp", mia_tp)
mlflow.log_metric("mia_fn", mia_fn)
mlflow.log_metric("forgetting_rate", forgetting_rate)

mlflow.end_run()
