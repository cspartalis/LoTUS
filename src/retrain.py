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
mlflow.set_tracking_uri(args.tracking_uri)
original_run = mlflow.get_run(args.run_id)

dataset = run.data.params["dataset"]
