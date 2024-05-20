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

from eval import log_l2_params_distance
from mlflow_utils import mlflow_tracking_uri

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

unlearn_run_id = "24936c3d118b43c3a6f1475ff9d3132b"

print("Using device:", DEVICE)
# Start MLflow run
mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.start_run(unlearn_run_id)
unlearn_run = mlflow.get_run(unlearn_run_id)
retrain_run_id = unlearn_run.data.params["reference_run_id"]
retrain_run = mlflow.get_run(retrain_run_id)

print("Loading unlearned model")
unlearned_model = mlflow.pytorch.load_model(
    f"{unlearn_run.info.artifact_uri}/unlearned_model"
)

unlearned_model.to(DEVICE)

print("Loading retrained model")
# retrained_model = mlflow.pytorch.load_model(f"{retrain_run.info.artifact_uri}/retrained_model")
retrained_model = mlflow.pytorch.load_model(f"{retrain_run.info.artifact_uri}/models")

retrained_model.to(DEVICE)

print("Loading verification error")
log_l2_params_distance(unlearned_model, retrained_model)
mlflow.end_run()
