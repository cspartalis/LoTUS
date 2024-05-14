import logging

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

logging.basicConfig(
    filename="extra_eval_debug.log",
    encoding="utf-8",
    level=logging.DEBUG,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)

from config import set_config
from data_utils import UnlearningDataLoader
from mlflow_utils import mlflow_tracking_uri
from models import ResNet18, ViT
from seed import set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print("Using device:", DEVICE)
    args = set_config()
    if args.run_id is None:
        raise ValueError("Please provide a run_id")

    # Start MLflow run
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    unlearn_run_id = args.run_id
    mlflow.start_run(unlearn_run_id)
    unlearn_run = mlflow.get_run(unlearn_run_id)

    retrain_run_id = unlearn_run.data.params["reference_run_id"]
    retrain_run = mlflow.get_run(retrain_run_id)

    seed = int(unlearn_run.data.params["seed"])
    dataset = unlearn_run.data.params["dataset"]
    batch_size = int(unlearn_run.data.params["batch_size"])
    model_str = unlearn_run.data.params["model"]
    method = unlearn_run.data.params["method"]
    logging.info(f"method: {method}")

    is_class_unlearning = retrain_run.data.params["is_class_unlearning"]
    is_class_unlearning = is_class_unlearning.lower() == "true"
    class_to_forget = retrain_run.data.params["class_to_forget"]

    print("Loading unlearned model")
    unlearned_model = mlflow.pytorch.load_model(
        f"{unlearn_run.info.artifact_uri}/unlearned_model"
    )
    unlearned_model.to(DEVICE)

    print("Loading retrained model")
    retrained_model = mlflow.pytorch.load_model(
        f"{retrain_run.info.artifact_uri}/retrained_model"
    )
    retrained_model.to(DEVICE)

    original_model = mlflow.pytorch.load_model(
        f"{retrain_run.info.artifact_uri}/original_model"
    )
    original_model.to(DEVICE)

    # Load data
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
    else:
        raise ValueError("Model not supported")

    retain_loader = dl["retain"]
    forget_loader = dl["forget"]
    test_loader = dl["test"]
    val_loader = dl["val"]

    # Evaluate unlearned model on MIA
    mia_results_unlearned = get_membership_attack_prob(
        retain_loader, forget_loader, test_loader, val_loader, unlearned_model
    )
    logging.info(f"mia_results_unlearned: {mia_results_unlearned}")

    # Evaluate retrained model on MIA
    mia_results_retrained = get_membership_attack_prob(
        retain_loader, forget_loader, test_loader, val_loader, retrained_model
    )
    logging.info(f"mia_results_retrained: {mia_results_retrained}")

    # Evaluate original model on MIA
    mia_results_original = get_membership_attack_prob(
        retain_loader, forget_loader, test_loader, val_loader, original_model
    )
    logging.info(f"mia_results_original: {mia_results_original}")

    mlflow.end_run()


def entropy(p, dim=-1, keepdim=False):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)


def collect_prob(data_loader, model):
    data_loader = torch.utils.data.DataLoader(
        data_loader.dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        prefetch_factor=10,
    )
    prob = []
    with torch.no_grad():
        for batch in data_loader:
            batch = [tensor.to(next(model.parameters()).device) for tensor in batch]
            # data, _, target = batch
            data, target = batch
            output = model(data)
            prob.append(F.softmax(output, dim=-1).data)
    return torch.cat(prob)


def get_membership_attack_data(
    retain_loader, forget_loader, test_loader, val_loader, model
):
    retain_prob = collect_prob(retain_loader, model)
    forget_prob = collect_prob(forget_loader, model)
    test_prob = collect_prob(test_loader, model)
    val_prob = collect_prob(val_loader, model)

    X_r = (
        torch.cat([entropy(retain_prob), entropy(test_prob), entropy(val_prob)])
        .cpu()
        .numpy()
        .reshape(-1, 1)
    )
    Y_r = np.concatenate(
        [np.ones(len(retain_prob)), np.zeros(len(test_prob)), np.zeros(len(val_prob))]
    )

    X_f = entropy(forget_prob).cpu().numpy().reshape(-1, 1)
    logging.info(f"X_f: {X_f}")
    Y_f = np.concatenate([np.ones(len(forget_prob))])
    return X_f, Y_f, X_r, Y_r


def get_membership_attack_prob(
    retain_loader, forget_loader, test_loader, val_loader, model
):
    logging.info("Collecting data for MIA...")
    X_f, Y_f, X_r, Y_r = get_membership_attack_data(
        retain_loader, forget_loader, test_loader, val_loader, model
    )
    # clf = SVC(C=3, gamma="auto", kernel="rbf")
    clf = LogisticRegression(
        class_weight="balanced", solver="lbfgs", multi_class="multinomial"
    )
    logging.info("Training MIA model...")
    clf.fit(X_r, Y_r)
    results = clf.predict(X_f)
    return results.mean()


main()
