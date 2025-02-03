"""
This script performs the unlearning process 
"""

import copy
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

from helpers.config import set_config
from helpers.data_utils import UnlearningDataLoader
from helpers.eval import compute_accuracy, log_js_proxy, log_mia, log_js
from helpers.mlflow_utils import mlflow_tracking_uri
from helpers.models import ResNet18, ViT
from helpers.seed import set_seed
from unlearning_methods.unlearning_base_class import UnlearningBaseClass

# pylint: enable=import-error
# ==== SETUP ====

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
("Using device:", DEVICE)
args = set_config()

# print(f"\n\t{args.method} - {args.registered_model}")

# Start MLflow run
now = datetime.now()
str_now = now.strftime("%m-%d-%H-%M")
mlflow.set_tracking_uri(mlflow_tracking_uri)

registered_model = args.registered_model
version = "latest"
try:
    retrained_model = mlflow.pytorch.load_model(
        model_uri=f"models:/{registered_model}/{version}"
    )
except:
    raise ValueError(f"Model {registered_model} not found")

_ = mlflow.pyfunc.load_model(model_uri=f"models:/{registered_model}/{version}")
retrained_run_id = _.metadata.run_id

# Load params from retraining run
retrain_run = mlflow.get_run(retrained_run_id)
seed = int(retrain_run.data.params["seed"])
dataset = retrain_run.data.params["dataset"]
model_str = retrain_run.data.params["model"]
is_class_unlearning = retrain_run.data.params["is_class_unlearning"]
is_class_unlearning = is_class_unlearning.lower() == "true"
class_to_forget = retrain_run.data.params["class_to_forget"]

# Set batch size via command line
batch_size = args.batch_size

# Load params from config
epochs = args.epochs
set_seed(seed, args.cudnn)

# Log parameters
if is_class_unlearning:
    mlflow.set_experiment(f"cs_{model_str}_{class_to_forget}")
else:
    mlflow.set_experiment(f"cs_{model_str}_{dataset}")

mlflow.start_run(run_name=f"{args.method}")
mlflow.log_param("datetime", str_now)
mlflow.log_param("reference_run_name", retrain_run.info.run_name)
mlflow.log_param("reference_run_id", retrained_run_id)
mlflow.log_param("seed", seed)
mlflow.log_param("cudnn", args.cudnn)
mlflow.log_param("dataset", dataset)
mlflow.log_param("model", model_str)
mlflow.log_param("batch_size", batch_size)
mlflow.log_param("epochs", args.epochs)
if args.using_CIFAKE:
    mlflow.log_param("method", f"{args.method}_cifake")
else:
    mlflow.log_param("method", args.method)


commit_hash = (
    subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
)
mlflow.log_param("git_commit_hash", commit_hash)

# Load data
is_vit = True if model_str == "vit" else False
if is_vit:
    image_size = 224
elif dataset == "mufac":
    image_size = 128
elif dataset == "tiny-imagenet":
    image_size = 64
else:  # cifar-10, cifar-100
    image_size = 32

UDL = UnlearningDataLoader(
    dataset,
    batch_size,
    image_size,
    seed,
    is_vit=is_vit,
    is_class_unlearning=is_class_unlearning,
    class_to_forget=class_to_forget,
    using_CIFAKE=args.using_CIFAKE,
)
dl, _ = UDL.load_data()
num_classes = len(UDL.classes)

# Load the original model
original_model = mlflow.pytorch.load_model(
    f"{retrain_run.info.artifact_uri}/original_model"
)
original_model.to(DEVICE)

original_eval = copy.deepcopy(original_model)

# ==== UNLEARNING ====
uc = UnlearningBaseClass(
    dl, batch_size, num_classes, original_model, epochs, dataset, seed
)

match args.method:
    case "finetune":
        from unlearning_methods.naive_unlearning_class import NaiveUnlearning

        naive_unlearning = NaiveUnlearning(uc)
        model, run_time = naive_unlearning.finetune()
    case "neggrad":
        from unlearning_methods.naive_unlearning_class import NaiveUnlearning

        naive_unlearning = NaiveUnlearning(uc)
        model, run_time = naive_unlearning.neggrad()
    case "relabel":
        from unlearning_methods.naive_unlearning_class import NaiveUnlearning

        naive_unlearning = NaiveUnlearning(uc)
        model, run_time = naive_unlearning.relabel()
    case "unsir":
        from unlearning_methods.unsir_class import UNSIR

        unsir = UNSIR(uc)
        model, run_time = unsir.unlearn()
    case "scrub":
        from unlearning_methods.scrub_class import SCRUB

        scrub = SCRUB(uc)
        model, run_time = scrub.unlearn()
    case "ssd":
        from unlearning_methods.ssd_class import SSD

        ssd = SSD(uc)
        model, run_time = ssd.unlearn()
    case "badT":
        from unlearning_methods.bad_teaching_class import BadTUnlearning

        badT = BadTUnlearning(uc)
        model, run_time = badT.unlearn()
    case "our":
        from unlearning_methods.lotus_class import LoTUS

        mlflow.log_param("Dr_subset_size", args.subset_size)

        maximize_entropy = LoTUS(uc)
        model, run_time = maximize_entropy.unlearn(
            subset_size=args.subset_size,
            is_class_unlearning=is_class_unlearning,
        )
    case "salun_relabel":
        from unlearning_methods.salun_class import SalUn

        salun = SalUn(uc)
        model, run_time = salun.unlearn(
            unlearning_method="relabel", is_class_unlearning=is_class_unlearning
        )
    case "salun_lotus":
        from unlearning_methods.salun_class import SalUn

        salun = SalUn(uc)
        model, run_time = salun.unlearn(unlearning_method="lotus", is_class_unlearning=is_class_unlearning)

# mlflow.pytorch.log_model(model, "unlearned_model")

# ==== EVALUATION =====
mlflow.log_metric("t", round(run_time, 2))

mia = log_mia(dl["retain"], dl["forget"], dl["test"], dl["val"], model)

js_proxy = log_js_proxy(
    unlearned=model,
    original=original_eval,
    forget_dl=dl["forget"],
    test_dl=dl["test"],
)

js = log_js(unlearned=model, gold_model=retrained_model, forget_dl=dl["forget"])

acc_retain = compute_accuracy(model, dl["retain"])
mlflow.log_metric("acc_retain", acc_retain)

acc_forget = compute_accuracy(model, dl["forget"])
mlflow.log_metric("acc_forget", acc_forget)

acc_test = compute_accuracy(model, dl["test"])
mlflow.log_metric("acc_test", acc_test)

acc_val = compute_accuracy(model, dl["val"])
mlflow.log_metric("acc_val", acc_val)

mlflow.end_run()

results_dict = {
    "mia": mia,
    "js": js,
    "forget": acc_forget,
    "retain": acc_retain,
    "run_time": run_time,
    "val": acc_val,
    "test": acc_test,
}
print(results_dict)
