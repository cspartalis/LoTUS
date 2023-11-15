"""
This script trains a deep learning model on a given dataset using PyTorch.
The model can be one of the following: ResNet18, AllCNN, or LiteViT.
The dataset can be one of the following: CIFAR10, CIFAR100, or Tiny-ImageNet.
The training can be done from scratch or by retraining without the forget set.
The script logs the training process and the evaluation metrics using MLflow.
The best model is saved as a PyTorch model and checkpoints are saved during training.
Early stopping can be enabled to stop training when the validation loss does not improve for a given number of epochs.
"""
import time
from datetime import datetime

import mlflow
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm  # pylint disable=import-error

from config import set_config  # pylint: disable=import-error
from data_utils import UnlearningDataLoader  # pylint: disable=import-error
from eval import compute_accuracy, mia  # pylint: disable=import-error
from models import AllCNN, ResNet18, VGG19  # pylint: disable=import-error
from seed import set_seed  # pylint: disable=import-error

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)
args = set_config()
set_seed(args.seed, args.cudnn)

# Start MLflow run
now = datetime.now()
str_now = now.strftime("%m-%d-%H-%M")
mlflow.set_tracking_uri("http://195.251.117.224:5000/")
mlflow.set_experiment(f"{args.model}_{args.dataset}")
mlflow.start_run(run_name=f"{args.model}_{args.dataset}_{args.train}_{str_now}")

# Log parameters
mlflow.log_param("seed", args.seed)
mlflow.log_param("cudnn", args.cudnn)
mlflow.log_param("dataset", args.dataset)
mlflow.log_param("model", args.model)
mlflow.log_param("train", args.train)
mlflow.log_param("batch_size", args.batch_size)
mlflow.log_param("epochs", args.epochs)
mlflow.log_param("loss", args.loss)
mlflow.log_param("optimizer", args.optimizer)
mlflow.log_param("lr", args.lr)
mlflow.log_param("momentum", args.momentum)
mlflow.log_param("weight_decay", args.weight_decay)
mlflow.log_param("lr_scheduler", args.lr_scheduler)
mlflow.log_param("scheduler_step1", args.scheduler_step1)
mlflow.log_param("scheduler_step2", args.scheduler_step2)
mlflow.log_param("scheduler_gamma", args.scheduler_gamma)
mlflow.log_param("early_stopping", args.early_stopping)

# Load data
UDL = UnlearningDataLoader(args.dataset, args.batch_size, args.seed)
dl, _ = UDL.load_data()
num_classes = len(UDL.classes)
input_channels = UDL.input_channels
image_size = UDL.image_size
if args.train == "original":
    train_loader = dl["train"]
    samples_per_class = UDL.get_samples_per_class("train")
elif args.train == "retrain":
    train_loader = dl["retain"]
    samples_per_class = UDL.get_samples_per_class("retain")
else:
    raise ValueError(
        "Use 'standard' to train from scratch or 'retraining' to retrain without the forget set"
    )

# Load model
if args.model == "resnet18":
    model = ResNet18(input_channels, num_classes)
elif args.model == "allcnn":
    model = AllCNN(input_channels, num_classes)
elif args.model == "vgg19":
    model = VGG19(input_channels, num_classes)
else:
    raise ValueError("Model not supported")


# Define loss function
if args.loss == "cross_entropy":
    loss_fn = nn.CrossEntropyLoss()
elif args.loss == "weighted_cross_entropy":
    l_samples_per_class = list(samples_per_class.values())
    total_samples = sum(l_samples_per_class)
    class_weights = [
        total_samples / (num_classes * samples_per_class[i]) for i in range(num_classes)
    ]
    class_weights = torch.FloatTensor(class_weights).to(
        DEVICE
    )  # pylint: disable=invalid-name
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

if args.lr_scheduler == "step":
    lr_scheduler = MultiStepLR(
        optimizer, milestones=[args.scheduler_step1, args.scheduler_step2], gamma=args.scheduler_gamma
    )
elif args.lr_scheduler == "none":
    lr_scheduler = None
else:
    raise ValueError("Learning rate scheduler not supported")


# Train model
model.to(DEVICE)
best_val_loss = float("inf")
start_time = time.time()
for epoch in tqdm(range(args.epochs)):
    model.train()
    train_loss = 0  # pylint: disable=invalid-name
    for inputs, targets in train_loader:
        inputs = inputs.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

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

    if args.lr_scheduler != "none":
        lr_scheduler.step()
    
    # Log metrics
    mlflow.log_metric("train_loss", train_loss, step=epoch)
    mlflow.log_metric("val_loss", val_loss, step=epoch)

    if args.early_stopping:
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            best_epoch = epoch
            best_time = (time.time() - start_time) / 60  # in minutes
            epochs_no_improve = 0  # pylint: disable=invalid-name
        else:
            epochs_no_improve += 1
            if epochs_no_improve == args.early_stopping:
                last_epoch = epoch
                break

# Save best model
if args.early_stopping:
    model.load_state_dict(best_model)
mlflow.pytorch.log_model(model, f"{args.model}_{args.dataset}_{args.train}_{str_now}")
mlflow.pytorch.save_model(
    model, f"checkpoints/{args.model}_{args.dataset}_{args.train}_{str_now}.pth"
)
print(f"Epoch {last_epoch}")
print(f"Best epoch {best_epoch}")

# Evaluation
# Epochs and time
mlflow.log_metric("best_epoch", best_epoch)
best_time = round(best_time, 2)
mlflow.log_metric("best_time", best_time)

# Accuracies
retain_accuracy = compute_accuracy(model, dl["retain"])
forget_accuarcy = compute_accuracy(model, dl["forget"])
test_accuracy = compute_accuracy(model, dl["test"])
val_accuracy = compute_accuracy(model, dl["val"])

retain_accuracy = round(retain_accuracy, 2)
forget_accuarcy = round(forget_accuarcy, 2)
test_accuracy = round(test_accuracy, 2)
val_accuracy = round(val_accuracy, 2)

mlflow.log_metric("retain_accuracy", retain_accuracy)
mlflow.log_metric("forget_accuarcy", forget_accuarcy)
mlflow.log_metric("test_accuracy", test_accuracy)
mlflow.log_metric("val_accuracy", val_accuracy)

# MIA
# Get the last train loss of the best_model
original_tr_loss = 0  # pylint: disable=invalid-name
model.eval()
with torch.inference_mode():
    for inputs, targets in train_loader:
        inputs = inputs.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        original_tr_loss += loss.item()
    original_tr_loss /= len(train_loader)

mlflow.log_metric("original_tr_loss_threshold", original_tr_loss)

mia_balanced_acc, mia_trp, mia_fpr = mia(
    model, dl["forget"], dl["val"], threshold=original_tr_loss
)

mia_balanced_acc = round(mia_balanced_acc.item(), 2)
mia_trp = round(mia_trp.item(), 2)
mia_fpr = round(mia_fpr.item(), 2)

mlflow.log_metric("mia_balanced_acc", mia_balanced_acc)
mlflow.log_metric("mia_trp", mia_trp)
mlflow.log_metric("mia_fpr", mia_fpr)


# End MLflow run
mlflow.end_run()
