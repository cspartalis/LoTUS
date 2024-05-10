"""
This script performs the unlearning process 
"""

import logging

# pylint: disable=import-error
import subprocess
import time
import warnings
from datetime import datetime

import mlflow
import torch
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR

from config import set_config
from data_utils import UnlearningDataLoader
from eval import (
    compute_accuracy,
    log_js_div,
    log_l2_params_distance,
    log_membership_attack_prob,
    log_zrf,
)
from mlflow_utils import mlflow_tracking_uri
from models import ResNet18, ViT
from seed import set_seed
from unlearning_base_class import UnlearningBaseClass

logger = logging.getLogger(__name__)

# pylint: enable=import-error

# ==== SETUP ====

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
("Using device:", DEVICE)
args = set_config()

if args.run_id is None:
    raise ValueError("Please provide a run_id")

# Start MLflow run
now = datetime.now()
str_now = now.strftime("%m-%d-%H-%M")
mlflow.set_tracking_uri(mlflow_tracking_uri)

# Load params from retraining run
retrain_run = mlflow.get_run(args.run_id)
seed = int(retrain_run.data.params["seed"])
dataset = retrain_run.data.params["dataset"]
model_str = retrain_run.data.params["model"]
epochs_to_retrain = int(retrain_run.data.metrics["best_epoch"])
optimizer_str = retrain_run.data.params["optimizer"]
momentum = float(retrain_run.data.params["momentum"])
weight_decay = float(retrain_run.data.params["weight_decay"])
is_class_unlearning = retrain_run.data.params["is_class_unlearning"]
is_class_unlearning = is_class_unlearning.lower() == "true"
class_to_forget = retrain_run.data.params["class_to_forget"]

# Set batch size via command line
batch_size = args.batch_size

# Load params from config
lr = args.lr
epochs = args.epochs
set_seed(seed, args.cudnn)

# Log parameters
if is_class_unlearning:
    mlflow.set_experiment(f"_{model_str}_{dataset}_{class_to_forget}_{seed}")
else:
    mlflow.set_experiment(f"_{model_str}_{dataset}_{seed}")

mlflow.start_run(run_name=f"{args.mu_method}")
mlflow.log_param("datetime", str_now)
mlflow.log_param("reference_run_name", retrain_run.info.run_name)
mlflow.log_param("reference_run_id", args.run_id)
mlflow.log_param("seed", seed)
mlflow.log_param("cudnn", args.cudnn)
mlflow.log_param("dataset", dataset)
mlflow.log_param("model", model_str)
mlflow.log_param("batch_size", batch_size)
mlflow.log_param("epochs", args.epochs)
mlflow.log_param("mu_method", args.mu_method)

commit_hash = (
    subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
)
mlflow.log_param("git_commit_hash", commit_hash)


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
# Load the original model
original_model = mlflow.pytorch.load_model(
    f"{retrain_run.info.artifact_uri}/original_model"
)
original_model.to(DEVICE)

# ==== UNLEARNING ====
if args.mu_method == "relabel_advanced":
    dl_start_prep_time = time.time()
    dl["mock_forget"] = UDL.get_mock_forget_dataloader(original_model)
    dl_prep_time = (time.time() - dl_start_prep_time) / 60  # in minute

uc = UnlearningBaseClass(
    dl, batch_size, num_classes, original_model, epochs, dataset, seed
)

match args.mu_method:
    case "finetune":
        from naive_unlearning_class import NaiveUnlearning

        naive_unlearning = NaiveUnlearning(uc)
        model, run_time = naive_unlearning.finetune()
    case "neggrad":
        from naive_unlearning_class import NaiveUnlearning

        naive_unlearning = NaiveUnlearning(uc)
        model, run_time = naive_unlearning.neggrad()
    case "amnesiac":
        from naive_unlearning_class import NaiveUnlearning

        naive_unlearning = NaiveUnlearning(uc)
        model, run_time = naive_unlearning.relabel()
    case "boundary":
        from boundary_unlearning_class import BoundaryUnlearning

        boundary_shrink = BoundaryUnlearning(uc)
        model, run_time = boundary_shrink.unlearn()
    case "unsir":
        from unsir_unlearning_class import UNSIR

        unsir = UNSIR(uc)
        model, run_time = unsir.unlearn()
    case "scrub":
        from scrub_unlearning_class import SCRUB

        scrub = SCRUB(uc)
        model, run_time = scrub.unlearn()
    case "ssd":
        from ssd_unlearning_class import SSD

        ssd = SSD(uc)
        model, run_time = ssd.unlearn()
    case "bad-teaching":
        from blindspot_unlearning_class import BlindspotUnlearning

        blindspot = BlindspotUnlearning(uc, unlearning_teacher=model, seed=seed)
        model, run_time = blindspot.unlearn()
    case "musum":
        from maximize_entropy_class import MaximizeEntropy

        mlflow.log_param("is_zapping", args.is_zapping)
        mlflow.log_param("is_once", args.is_once)
        mlflow.log_param("forget_loss", args.forget_loss)
        mlflow.log_param("Dr_subset_size", args.subset_size)

        maximize_entropy = MaximizeEntropy(uc)
        model, run_time = maximize_entropy.unlearn(
            is_zapping=args.is_zapping,
            is_once=args.is_once,
            str_forget_loss=args.forget_loss,
            subset_size=args.subset_size,
        )

mlflow.pytorch.log_model(model, "unlearned_model")

# ==== EVALUATION =====

# Compute accuracy on the test dataset
is_multi_label = True if dataset == "mucac" else False
acc_test = compute_accuracy(model, dl["test"], is_multi_label)
mlflow.log_metric("acc_test", acc_test)

# Load the retrained model (is needed for js_div, l2_params_distance, and mia)
retrained_model = mlflow.pytorch.load_model(
    f"{retrain_run.info.artifact_uri}/retrained_model"
)

if args.mu_method != "musum":
    log_membership_attack_prob(dl["retain"], dl["forget"], dl["test"], dl["val"], model)

log_js_div(retrained_model, model, dl["train"], dataset)

log_zrf(model, retrained_model, dl["forget"], is_multi_label)

mlflow.log_metric("t", run_time)

mlflow.end_run()
