# pylint: disable=import-error
import time
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
    get_l2_weight_distance,
    mia,
)
from mlflow_utils import mlflow_tracking_uri
from models import VGG19, AllCNN, ResNet18
from seed import set_seed
# pylint: enable=import-error

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

# Load params from retrained model (the same as the original model)
seed = int(retrain_run.data.params["seed"])
model_str = retrain_run.data.params["model"]
dataset = retrain_run.data.params["dataset"]
set_seed(seed, args.cudnn)

# Log parameters
mlflow.set_experiment(f"{args.model}_{args.dataset}")
mlflow.start_run(run_name=f"{model_str}_{dataset_str}_finetune_{str_now}")
mlflow.log_param("seed", seed)
mlflow.log_param("cudnn", args.cudnn)
mlflow.log_param("dataset", model_str)
mlflow.log_param("model", model_str)
mlflow.log_param("batch_size", args.batch_size)
mlflow.log_param("epochs", args.epochs)
mlflow.log_param("loss", args.loss)
mlflow.log_param("optimizer", args.optimizer)
mlflow.log_param("lr", args.lr)
mlflow.log_param("momentum", args.momentum)
mlflow.log_param("weight_decay", args.weight_decay)
mlflow.log_param("warmup_epochs", args.warmup_epochs)
mlflow.log_param("early_stopping", args.early_stopping)

# Load data
UDL = UnlearningDataLoader(dataset, args.batch_size, seed)
dl, _ = UDL.load_data()
num_classes = len(UDL.classes)
input_channels = UDL.input_channels
image_size = UDL.image_size

# Load the original model
model = mlflow.pyfunc.load_model(
    f"runs:/{args.run_id}/artifacts/original_model"
)

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
    class_weights = torch.FloatTensor(class_weights).to(DEVICE)  # pylint: disable=invalid-name
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

# Set optimizer
if args.optimizer == "sgd":
    optimizer = SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
elif args.optimizer == "adam":
    optimizer = Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
else:
    raise ValueError("Optimizer not supported")

# Set learning rate scheduler
# fmt: off
lr_lambda = lambda epoch: min(1.0, (epoch + 1) / args.warmup_epochs) * (1.0 - max(0.0, (epoch + 1) - args.warmup_epochs) / (args.epochs - args.warmup_epochs))  # pylint: disable=line-too-long
# fmt: on
lr_scheduler = LambdaLR(optimizer, lr_lambda)

# Train the model
model.to(DEVICE)
start_time = time.time()
for epoch in tqdm(range(epochs)):
    model.train()
    for inputs, targets in dl["retain"]:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    lr_scheduler.step()

    acc_retain = compute_accuracy(model, dl["retain"], DEVICE)
    acc_forget = compute_accuracy(model, dl["forget"], DEVICE)
    acc_test = compute_accuracy(model, dl["test"], DEVICE)
    time = time.time() - start_time

    # Log accuracies and time
    mlflow.log_metric("acc_retain", acc_retain, step=epoch)
    mlflow.log_metric("acc_forget", acc_forget, step=epoch)
    mlflow.log_metric("acc_test", acc_test, step=epoch)
    mlflow.log_metric("time", time, step=epoch)

# Evaluation 