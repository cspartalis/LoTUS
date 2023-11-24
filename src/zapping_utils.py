import os

import matplotlib.pyplot as plt
import mlflow
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def zapping(model, weight_mask) -> None:
    """
    This function resets the weights of the last fully connected layer of a neural network.
    If the_class is None, then all the weights of the fc layer are reset.
    If the_class is an integer, then only the weights corresponding to the the_class the_class are reset.
    Args:
        model (torch.nn.Module): The model to unlearn.
        weight_mask (torch.Tensor): It contains ones for the weights to be zapped, zeros for the others.
    """
    fc_layer = model.get_last_fc_layer()
    # Get the weights of the fc layer
    weights_reset = fc_layer.weight.data.detach().clone()
    # Reset the weights corresponding the the_class i (fc --> lr)
    torch.nn.init.xavier_normal_(tensor=weights_reset, gain=1.0)
    # torch.nn.init.kaiming_normal_(tensor=weights_reset, mode="fan_out", nonlinearity="relu")
    # torch.nn.init.orthogonal_(tensor=weights_reset, gain=1.0)
    # torch.nn.init.uniform_(tensor=weights_reset, a=-0.1, b=0.1)

    # Reset the weights of the fc layer based on the mask
    fc_layer.weight.data[weight_mask == 1] = weights_reset[weight_mask == 1]


def count_fc_parameters(model):
    """
    This function computes the number of parameters in the fully connected (fc) layer of a model.
    """
    fc_layer = model.get_last_fc_layer()
    num_parameters = torch.numel(fc_layer.weight)
    bias_parameters = torch.numel(fc_layer.bias)
    return num_parameters, bias_parameters


def get_fc_gradients(model, dataloader, loss_fn):
    """
    This function computes and stores the gradients of the weights in the fully connected layer
    of the given model after a forward pass using the provided dataloader.
    """
    # Set the model to training mode
    model.train()

    fc_layer = model.get_last_fc_layer()

    # Create an empty list to store the gradients
    grads = []

    # Perform a forward pass and backward pass of the model
    for inputs, targets in dataloader:
        model.zero_grad()
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        for name, param in fc_layer.named_parameters():
            if "weight" in name:
                grads.append(param.grad.detach().clone())

    # Concatenate the grads
    stacked_grads = torch.stack(grads)

    # Compute the means and std of the grads for each neuron
    fc_grads_mean = torch.mean(stacked_grads, dim=0)

    # Take the absolute value of the grads
    fc_grads_mean_abs = torch.abs(fc_grads_mean)

    # Min-max normalize the grads
    fc_grads_mean_abs_norm = (fc_grads_mean_abs - fc_grads_mean_abs.min()) / (
        fc_grads_mean_abs.max() - fc_grads_mean_abs.min()
    )

    return fc_grads_mean_abs_norm


def visualize_fc_grads(fc_grads, filename):
    """
    Visualizes a tensor as a heatmap.

    Args:
        fc_grads (torch.Tensor): Absolute of the normalized gradients of the weights in the fully connected layer.

    Returns:
        None
    """
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create the "plots" directory one level up
    plots_dir = os.path.join(os.path.dirname(__file__), "..", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    heatmap = ax.imshow(fc_grads.cpu(), cmap="viridis", aspect="auto")

    # Add colorbar
    fig.colorbar(heatmap)

    # Set the title and labels
    ax.set_xlabel("Weights")
    ax.set_ylabel("Neurons")

    # Save the figure
    mlflow.log_figure(fig, filename + ".png")


def get_diff_gradients(forget_grads, retain_grads):
    """
    This function computes the difference between the gradients of the weights in the fully connected layer
    of the given model after a forward pass using the provided dataloader.
    """
    # Compute the difference between the gradients
    diff_grads = forget_grads - retain_grads

    # Min-max normalize the grads
    diff_grads_norm = (diff_grads - diff_grads.min()) / (
        diff_grads.max() - diff_grads.min()
    )

    return diff_grads_norm


def get_weight_mask(fc_gradients, threshold=0):
    """
    This function computes a mask for the weights in the fully connected layer
    of the given model after a forward pass using the provided dataloader.

    Args:
        fc_gradients (torch.Tensor): The gradients of the weights in the fully connected layer. Value range: [0, 1].
        threshold (float): The threshold for the gradients.

    Returns:
        weight_mask (torch.Tensor): The mask for the weights in the fully connected layer.
    """
    # Create a mask for the weights
    weight_mask = torch.zeros_like(fc_gradients)

    # Set the weights to 1 if the gradient is greater than the threshold
    weight_mask[fc_gradients > threshold] = 1

    return weight_mask
