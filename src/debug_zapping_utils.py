# pylint: disable=import-error
import torch
from models import ResNet18, VGG19
from data_utils import UnlearningDataLoader
from zapping_utils import get_fc_activations, count_fc_parameters, get_fc_gradients, visualize_tensor
import numpy as np
# pylint: enable=import-error

# Load data
UDL = UnlearningDataLoader("cifar-10", 128, 3407)
dl, _ = UDL.load_data()
num_classes = len(UDL.classes)
input_channels = UDL.input_channels
image_size = UDL.image_size

model = ResNet18(input_channels, num_classes)
model.to("cuda")

grads_forget = get_fc_gradients(model, dl["forget"])
visualize_tensor(grads_forget, "Gradients in the FC layer after training on the forget dataset")

grads_retain = get_fc_gradients(model, dl["retain"])
visualize_tensor(grads_retain, "Gradients in the FC layer after training on the retain dataset")

diff_grads = grads_forget - grads_retain


# Perform min-max normalization on diff_grads
min_val = torch.min(diff_grads)
max_val = torch.max(diff_grads)
normalized_diff_grads = (diff_grads - min_val) / (max_val - min_val)

visualize_tensor(normalized_diff_grads, "Normalized difference in the gradients in the FC layer")

# num_w, num_b = count_fc_parameters(model)
# print(f"Number of weights: {num_w}")
# print(f"Number of biases: {num_b}")

# activations = get_fc_activations(model, dl["forget"])
