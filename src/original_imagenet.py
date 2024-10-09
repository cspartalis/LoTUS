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
import warnings
from datetime import datetime

import mlflow
import torch
from config import set_config
from data_utils import UnlearningDataLoader
from eval import compute_accuracy_imagenet, log_js_proxy, log_mia
from mlflow_utils import mlflow_tracking_uri
import random
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
if args.is_class_unlearning:
    mlflow.set_experiment(
        f"_{args.model}_{args.dataset}_{args.class_to_forget}_{args.seed}"
    )
else:
    mlflow.set_experiment(f"_{args.model}_{args.dataset}_{args.seed}")
mlflow.start_run(run_name="original")


# Log parameters
mlflow.log_param("datetime", str_now)
mlflow.log_param("seed", args.seed)
mlflow.log_param("cudnn", args.cudnn)
mlflow.log_param("dataset", args.dataset)
mlflow.log_param("model", args.model)
mlflow.log_param("batch_size", args.batch_size)
mlflow.log_param("epochs", args.epochs)
commit_hash = (
    subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
)
mlflow.log_param("git_commit_hash", commit_hash)
mlflow.log_param("is_class_unlearning", args.is_class_unlearning)
mlflow.log_param("method", "original")

image_size = 224

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
print("Number of classes:", num_classes)

from torchvision.models import vit_b_16, ViT_B_16_Weights

model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
model = model.to(DEVICE)

mlflow.pytorch.log_model(model, "original_model")
mlflow.pytorch.log_model(
    model,
    artifact_path="models",
    registered_model_name=f"{args.model}-{args.dataset}-{args.seed}-original",
)


model.eval()
acc_retain = compute_accuracy_imagenet(model, dl["retain"])
mlflow.log_metric("acc_retain", acc_retain)
acc_forget = compute_accuracy_imagenet(model, dl["forget"])
mlflow.log_metric("acc_forget", acc_forget)
acc_val = compute_accuracy_imagenet(model, dl["val"])
mlflow.log_metric("acc_val", acc_val)
acc_test = compute_accuracy_imagenet(model, dl["test"])
mlflow.log_metric("acc_test", acc_test)
js_div = log_js_proxy(model, model, dl["forget"], dl["test"])
mia = log_mia(dl["retain"], dl["forget"], dl["test"], dl["val"], model)
mlflow.end_run()

exit()
