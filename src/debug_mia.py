import mlflow.pytorch
import mlflow
import torch
from eval import mia
from data_utils import UnlearningDataLoader
from seed import set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0
RNG = set_seed(SEED)

# Load the model and its parameters

mlflow.set_tracking_uri("http://195.251.117.224:5000/")
model = mlflow.pytorch.load_model(
    "mlflow-artifacts:/121363690789578682/dbb594598551458db19401e78909c7f4/artifacts/allcnn_mnist_original_11-09-09-58"
)
# Get the logged parameters from mlflow and load the data
run = mlflow.get_run("dbb594598551458db19401e78909c7f4")

# Load the data
dataset = run.data.params["dataset"]
batch_size = int(run.data.params["batch_size"])
seed = int(run.data.params["seed"])
RNG = set_seed(seed)
UDL = UnlearningDataLoader(dataset, batch_size, seed)
dl, _ = UDL.load_data()
num_classes = len(UDL.classes)

criterion = torch.nn.CrossEntropyLoss()
model.to(DEVICE)
model.eval()
total_loss = 0
with torch.inference_mode():
    for inputs, targets in dl["train"]:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item()
    total_loss /= len(dl["train"])

# Compute the MIA metrics
mia_bacc, mia_tpr, mia_fpr = mia(
    model, dl["forget"], dl["val"], total_loss, num_classes
)
print(f"MIA metrics: {mia_bacc:.2f}, {mia_tpr:2.f}, {mia_fpr:.2f}")
