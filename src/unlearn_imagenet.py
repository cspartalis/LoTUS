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
from helpers.eval import (
    compute_accuracy,
    log_js_proxy,
)
from helpers.mlflow_utils import mlflow_tracking_uri
from helpers.models import ResNet18, ViT
from helpers.seed import set_seed
from unlearning_methods.unlearning_base_class import UnlearningBaseClass


# pylint: enable=import-error
def main():
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
    # No need to load the original model from mlflow
    # try:
    #     original_model = mlflow.pytorch.load_model(
    #         model_uri=f"models:/{registered_model}/{version}"
    #     )
    # except:
    #     raise ValueError(f"Model {registered_model} not found")

    _ = mlflow.pyfunc.load_model(model_uri=f"models:/{registered_model}/{version}")
    original_run_id = _.metadata.run_id

    # Load params from retraining run
    original_run = mlflow.get_run(original_run_id)
    seed = int(original_run.data.params["seed"])
    dataset = original_run.data.params["dataset"]
    model_str = original_run.data.params["model"]
    is_class_unlearning = original_run.data.params["is_class_unlearning"]
    is_class_unlearning = is_class_unlearning.lower() == "true"
    # class_to_forget = original_run.data.params["class_to_forget"]

    # Set batch size via command line
    batch_size = args.batch_size

    # Load params from helpers.config
    epochs = args.epochs
    set_seed(seed, args.cudnn)

    # Log parameters
    if is_class_unlearning:
        pass
        # mlflow.set_experiment(f"_{model_str}_{class_to_forget}_{seed}")
    else:
        mlflow.set_experiment(f"_{model_str}_{dataset}_{seed}")

    mlflow.start_run(run_name=f"{args.method}")
    mlflow.log_param("datetime", str_now)
    mlflow.log_param("reference_run_name", original_run.info.run_name)
    mlflow.log_param("reference_run_id", original_run_id)
    mlflow.log_param("seed", seed)
    mlflow.log_param("cudnn", args.cudnn)
    mlflow.log_param("dataset", dataset)
    mlflow.log_param("model", model_str)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("epochs", args.epochs)
    mlflow.log_param("method", args.method)

    commit_hash = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
    )
    mlflow.log_param("git_commit_hash", commit_hash)

    image_size = 224

    UDL = UnlearningDataLoader(
        dataset,
        batch_size,
        image_size,
        seed,
        is_vit=True,
        is_class_unlearning=is_class_unlearning,
        # class_to_forget=class_to_forget,
        class_to_forget=None,
    )
    dl, _ = UDL.load_data()
    num_classes = len(UDL.classes)

    from torchvision.models import vit_b_16, ViT_B_16_Weights

    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    model = model.to(DEVICE)
    original_model = copy.deepcopy(model)
    original_model.eval()

    # ==== UNLEARNING ====
    uc = UnlearningBaseClass(dl, batch_size, num_classes, model, epochs, dataset, seed)

    match args.method:
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
        case "unsir":
            from unsir_class import UNSIR

            unsir = UNSIR(uc)
            model, run_time = unsir.unlearn()
        case "scrub":
            from scrub_class import SCRUB

            scrub = SCRUB(uc)
            model, run_time = scrub.unlearn()
        case "ssd":
            from ssd_class import SSD

            ssd = SSD(uc)
            model, run_time = ssd.unlearn()
        case "bad-teacher":
            from bad_teaching_class import bad_teachingUnlearning

            bad_teaching = bad_teachingUnlearning(uc, unlearning_teacher=model)
            model, run_time = bad_teaching.unlearn()
        case "our":
            from our_class import Our

            branch_name = (
                subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
                .strip()
                .decode("utf-8")
            )
            mlflow.log_param("branch_name", branch_name)
            mlflow.log_param("Dr_subset_size", args.subset_size)

            maximize_entropy = Our(uc)
            model, run_time = maximize_entropy.unlearn(
                subset_size=args.subset_size,
                is_class_unlearning=is_class_unlearning,
            )

    mlflow.pytorch.log_model(model, "unlearned_model")

    # ==== EVALUATION =====
    acc_forget = compute_accuracy(model, dl["forget"], False)
    mlflow.log_metric("acc_forget", acc_forget)
    acc_retain = compute_accuracy(model, dl["retain"], False)
    mlflow.log_metric("acc_retain", acc_retain)
    acc_val = compute_accuracy(model, dl["val"], False)
    mlflow.log_metric("acc_val", acc_val)
    acc_test = compute_accuracy(model, dl["test"], False)
    mlflow.log_metric("acc_test", acc_test)
    js_proxy = log_js_proxy(
        unlearned=model, original=original_model, forget_dl=dl["forget"], test_dl=dl["test"]
    }

    run_time = round(run_time, 2)
    mlflow.log_metric("t", run_time)

    mlflow.end_run()

    results_dict = {
        "js_proxy": js_proxy,
        "acc_forget": acc_forget,
        "acc_retain": acc_retain,
        "run_time": run_time,
        "acc_val": acc_val,
        "acc_test": acc_test,
    }
    print(results_dict)

    return results_dict


if __name__ == "__main__":
    results_dict = main()
