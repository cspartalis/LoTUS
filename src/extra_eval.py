import logging

import mlflow
import torch
from sklearn.linear_model import LogisticRegression

from config import set_config
from data_utils import UnlearningDataLoader
from eval import compute_accuracy, get_membership_attack_data, log_l2_params_distance
from mlflow_utils import mlflow_tracking_uri

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

unlearn_run_id = "24936c3d118b43c3a6f1475ff9d3132b"

print("Using device:", DEVICE)
args = set_config()


def main():
    # Start mlflow
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    retrained_run_id = "16f0a4c834614b5a92c9d7e29fa9c9e1"
    original_metrics_class_unlearning(retrained_run_id)

    # unlearn_run_id = ""
    # verification_error(unlearn_run_id)


def original_metrics_class_unlearning(retrained_run_id):
    mlflow.start_run(retrained_run_id)
    retrain_run = mlflow.get_run(retrained_run_id)
    original_model = mlflow.pytorch.load_model(
        f"{retrain_run.info.artifact_uri}/original_model"
    )
    original_model.to(DEVICE)
    dataset = retrain_run.data.params["dataset"]
    seed = int(retrain_run.data.params["seed"])
    model_str = retrain_run.data.params["model"]
    batch_size = int(retrain_run.data.params["batch_size"])
    is_class_unlearning = retrain_run.data.params["is_class_unlearning"]
    is_class_unlearning = is_class_unlearning.lower() == "true"
    class_to_forget = retrain_run.data.params["class_to_forget"]

    # Load data
    if model_str == "resnet18":
        if dataset in ["cifar-10", "cifar-100"]:
            image_size = 32
        else:
            image_size = 128
        UDL = UnlearningDataLoader(
            dataset,
            batch_size,
            image_size,
            seed,
            is_vit=False,
            is_class_unlearning=is_class_unlearning,
            class_to_forget=class_to_forget,
        )
    else:
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

    forget_acc = compute_accuracy(original_model, dl["forget"])
    retain_acc = compute_accuracy(original_model, dl["forget"])

    mlflow.log_metric("original_forget_acc", forget_acc)
    mlflow.log_metric("original_retain_acc", retain_acc)

    # MIA prob
    X_f, Y_f, X_r, Y_r = get_membership_attack_data(
        dl["retain"], dl["forget"], dl["test"], dl["val"], original_model
    )

    clf = LogisticRegression(
        class_weight="balanced",
        solver="lbfgs",
        multi_class="multinomial",
        random_state=42,
    )

    clf.fit(X_r, Y_r)
    results = clf.predict(X_f)
    prob_mia = results.mean() * 100
    prob_mia = round(prob_mia, 2)
    mlflow.log_metric("original_MIA_prob", prob_mia)


def verification_error(unlearn_run_id: str):
    mlflow.start_run(unlearn_run_id)
    unlearn_run = mlflow.get_run(unlearn_run_id)
    retrain_run_id = unlearn_run.data.params["reference_run_id"]
    retrain_run = mlflow.get_run(retrain_run_id)

    unlearned_model = mlflow.pytorch.load_model(
        f"{unlearn_run.info.artifact_uri}/unlearned_model"
    )
    unlearned_model.to(DEVICE)

    print("Loading retrained model")
    retrained_model = mlflow.pytorch.load_model(
        f"{retrain_run.info.artifact_uri}/models"
    )
    retrained_model.to(DEVICE)

    log_l2_params_distance(unlearned_model, retrained_model)

    mlflow.end_run()


if __name__ == "__main__":

    main()
