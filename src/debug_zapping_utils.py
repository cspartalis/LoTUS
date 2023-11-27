# pylint: disable=import-error
import numpy as np
import torch

from data_utils import UnlearningDataLoader
from models import VGG19, ResNet18
import zapping_utils as zu

from data_utils import UnlearningDataLoader

# pylint: enable=import-error

# Load data
UDL = UnlearningDataLoader("cifar-10", 128, 3407)
dl, _ = UDL.load_data()
num_classes = len(UDL.classes)
input_channels = UDL.input_channels
image_size = UDL.image_size

model = ResNet18(input_channels, num_classes)
model.to("cuda")
dl["mixed"] = UDL.get_mixed_dataloader(model)

for inputs, targets in dl["mixed"]:
    print(inputs.shape)

grads_forget = zu.get_fc_gradients(model, dl["forget"])
# zu.visualize_fc_grads(grads_forget)

# grads_retain = zu.get_fc_gradients(model, dl["retain"])
# zu.visualize_fc_grads(grads_retain)

# diff_grads = grads_forget - grads_retain


# # Perform min-max normalization on diff_grads
# min_val = torch.min(diff_grads)
# max_val = torch.max(diff_grads)
# normalized_diff_grads = (diff_grads - min_val) / (max_val - min_val)

# zu.visualize_fc_grads(
#     normalized_diff_grads, "Normalized difference in the gradients in the FC layer"
# )

# num_w, num_b = count_fc_parameters(model)
# print(f"Number of weights: {num_w}")
# print(f"Number of biases: {num_b}")

# activations = get_fc_activations(model, dl["forget"])
