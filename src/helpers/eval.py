import logging

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from tqdm import tqdm

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_accuracy(model, dataloader):
    """
    Code from PyTorch Forum for ImageNet1k evaluation of accuracy
    """
    correct = 0
    total = 0
    model.to(DEVICE, non_blocking=True)
    model.eval()
    with torch.inference_mode():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = correct / total
    accuracy = round(accuracy, 2)
    return accuracy


def JSDiv(p, q, epsilon=1e-8):
    """
    Jensen-Shannon Divergence

    Args:
    p, q: PyTorch tensors representing probability distributions
    epsilon: small value to avoid log(0)

    Returns:
    JS divergence between p and q
    """
    # Ensure inputs are valid probability distributions
    p = torch.clamp(p, min=epsilon, max=1.0)
    q = torch.clamp(q, min=epsilon, max=1.0)

    # Normalize to ensure they sum to 1
    p = p / torch.sum(p)
    q = q / torch.sum(q)

    # Calculate midpoint distribution
    m = 0.5 * (p + q)

    # Calculate JS divergence
    return 0.5 * (
        F.kl_div(p.log(), m, reduction="batchmean")
        + F.kl_div(q.log(), m, reduction="batchmean")
    )


def log_js_proxy(unlearned, original, forget_dl, test_dl):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the mean class probabilities
    of two models (unlearned and original) on two datasets (forget_dl and test_dl)
    and logs the result using MLflow. Thsi metric can be used to evaluate unlearning
    effectiveness when the gold standard model (i.e., retrained from scratch only on retain data)
    is not available.

    Args:
        unlearned (torch.nn.Module): The model that has undergone the unlearning process.
        original (torch.nn.Module): The original model before unlearning.
        forget_dl (torch.utils.data.DataLoader): DataLoader for the dataset to be forgotten.
        test_dl (torch.utils.data.DataLoader): DataLoader for the test dataset.

    Returns:
        float: The computed Jensen-Shannon Divergence (JSD) between the mean class
               probabilities of the unlearned and original models.
    """

    def check_model_type(model):
        model_name = model.__class__.__name__.lower()
        if "resnet" in model_name:
            return "ResNet"
        elif "vit" in model_name:
            return "ViT"
        else:
            raise ValueError("Model not supported")
    
    # Control the batch size to maximize GPU utilization without running OOM.
    batch_size = 2048 if check_model_type(original) == "ResNet" else 512

    unlearned.eval()
    original.eval()
    forget_loader = torch.utils.data.DataLoader(
        forget_dl.dataset, batch_size=batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dl.dataset, batch_size=batch_size, shuffle=False
    )

    # Dictionary to store mean probabilities for each class in test_loader
    class_mean_probs_test = {}
    class_mean_probs_forget = {}

    # Compute mean probabilities for each class in test_loader
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="zrf - test loader"):
            x = x.to(DEVICE)
            probs = F.softmax(original(x), dim=1).detach().cpu()
            for i, label in enumerate(y):
                if label.item() not in class_mean_probs_test:
                    class_mean_probs_test[label.item()] = []
                class_mean_probs_test[label.item()].append(probs[i])

    # Compute the mean for each class
    for label in class_mean_probs_test:
        class_mean_probs_test[label] = torch.stack(class_mean_probs_test[label]).mean(
            dim=0
        )
        class_mean_probs_test[label] = (
            class_mean_probs_test[label] / class_mean_probs_test[label].sum()
        )  # Normalize to sum to 1

    with torch.no_grad():
        for x, y in tqdm(forget_loader, desc="zrf - forget loader"):
            x = x.to(DEVICE)
            probs = F.softmax(unlearned(x), dim=1).detach().cpu()
            for i, label in enumerate(y):
                if label.item() not in class_mean_probs_forget:
                    class_mean_probs_forget[label.item()] = []
                class_mean_probs_forget[label.item()].append(probs[i])

    # Compute the mean for each class
    for label in class_mean_probs_forget:
        class_mean_probs_forget[label] = torch.stack(
            class_mean_probs_forget[label]
        ).mean(dim=0)
        class_mean_probs_forget[label] = (
            class_mean_probs_forget[label] / class_mean_probs_forget[label].sum()
        )  # Normalize to sum to 1

    # Check for missing keys
    for key in range(1000):
        if key not in class_mean_probs_forget:
            class_mean_probs_test.pop(key, None)

    # Assert that the number of classes in both dictionaries is the same
    assert len(class_mean_probs_test) == len(
        class_mean_probs_forget
    ), "The number of classes in test_loader and forget_loader do not match."

    # Order both dictionaries by key
    class_mean_probs_test = dict(sorted(class_mean_probs_test.items()))
    class_mean_probs_forget = dict(sorted(class_mean_probs_forget.items()))

    js_div = JSDiv(
        torch.stack(list(class_mean_probs_forget.values())),
        torch.stack(list(class_mean_probs_test.values())),
    )
    js_div = round(js_div.item(), 6)
    mlflow.log_metric("js_proxy", js_div)
    return js_div


def log_js(unlearned, gold_model, forget_dl, step=None):
    model_preds = []
    gold_model_preds = []
    with torch.no_grad():
        for x, _ in forget_dl.dataset:
            x = x.unsqueeze(0).to(DEVICE)
            model_output = unlearned(x)
            gold_model_output = gold_model(x)
            model_preds.append(F.softmax(model_output, dim=1).detach().cpu())
            gold_model_preds.append(F.softmax(gold_model_output, dim=1).detach().cpu())

    model_preds = torch.cat(model_preds, axis=0)
    gold_model_preds = torch.cat(gold_model_preds, axis=0)
    js = JSDiv(model_preds, gold_model_preds)
    js = js.item()
    js = round(js, 6)
    mlflow.log_metric("js", js, step=step)
    return js


def log_zrf(tmodel, gold_model, forget_dl, step=None):
    """
    My Comment: The Zero-Retain Forgetting (ZRF) metric was introduced in "Can bad teaching induce forgetting?"
    by Chundawat et al. (2023). It is basically 1 - JSDiv between the predictions of the target model
    and the gold model on the forget dataset. So why not just use the JSDiv function?

    Computes and logs the Zero-Retain Forgetting (ZRF) metric using Jensen-Shannon Divergence (JSDiv)
    between the predictions of the target model and the gold model on the forget dataset.
    Args:
        tmodel (torch.nn.Module): The target model whose predictions are to be evaluated.
        gold_model (torch.nn.Module): The gold standard model for comparison.
        forget_dl (torch.utils.data.DataLoader): DataLoader containing the dataset to be evaluated.
        step (int, optional): The step or epoch number for logging purposes. Defaults to None.
    Returns:
        float: The computed ZRF value, rounded to four decimal places.
    """

    model_preds = []
    gold_model_preds = []
    with torch.no_grad():
        for x, _ in forget_dl.dataset:
            x = x.unsqueeze(0).to(DEVICE)
            model_output = tmodel(x)
            gold_model_output = gold_model(x)
            model_preds.append(F.softmax(model_output, dim=1).detach().cpu())
            gold_model_preds.append(F.softmax(gold_model_output, dim=1).detach().cpu())

    model_preds = torch.cat(model_preds, axis=0)
    gold_model_preds = torch.cat(gold_model_preds, axis=0)
    zrf = 1 - JSDiv(model_preds, gold_model_preds)
    zrf = zrf.item()
    zrf = round(zrf, 4)
    mlflow.log_metric("zrf", zrf, step=step)
    return zrf


# ========================================================================================
# The following functions is from the github repository: Selective Synaptic Dampening
# They have been used for evaluation MIA succes rate in by both:
# 1. Can bad teaching induce unlearning? by Chundawat et al. (2023)
# 2. Selective Synaptic Dampening by Foster et al. (2024)
# ===========================


def entropy(p, dim=-1, keepdim=False):
    """
    Computes the entropy of a probability distribution.
    Entropy is a measure of the uncertainty or randomness in a probability distribution.
    This function calculates the entropy for the given probability distribution `p`.
    Args:
        p (torch.Tensor): The input probability distribution tensor.
        dim (int, optional): The dimension along which the entropy is computed. Default is -1.
        keepdim (bool, optional): Whether to retain the reduced dimension in the output tensor. Default is False.
    Returns:
        torch.Tensor: The entropy of the input probability distribution.
    """

    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)


def collect_prob(data_loader, model):
    """
    Collects the probability distributions of the model's predictions for the given data.

    Args:
        data_loader (torch.utils.data.DataLoader): DataLoader containing the dataset to evaluate.
        model (torch.nn.Module): The model used to make predictions.

    Returns:
        torch.Tensor: A tensor containing the concatenated probability distributions of the model's predictions.
    """

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
            data, _ = batch
            output = model(data)
            prob.append(F.softmax(output, dim=-1).data)
    return torch.cat(prob)


def get_membership_attack_data(
    retain_loader, forget_loader, test_loader, val_loader, model, step=None
):
    """
    Collects and processes data for membership inference attacks.
    This function computes the entropy of the model's predictions on different datasets
    (retain, forget, test, and validation) and logs the mean entropy values using mlflow.
    It returns the feature and label arrays for both the forget and retain datasets.
    Args:
        retain_loader (DataLoader): DataLoader for the retain dataset.
        forget_loader (DataLoader): DataLoader for the forget dataset.
        test_loader (DataLoader): DataLoader for the test dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        model (torch.nn.Module): The model to evaluate.
        step (int, optional): The current step for logging metrics. Defaults to None.
    Returns:
        tuple: A tuple containing:
            - X_f (numpy.ndarray): Features for the forget dataset.
            - Y_f (numpy.ndarray): Labels for the forget dataset.
            - X_r (numpy.ndarray): Features for the retain dataset.
            - Y_r (numpy.ndarray): Labels for the retain dataset.
    """
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
    Y_f = np.concatenate([np.ones(len(forget_prob))])

    # Log Mean Entropy
    # retain_entorpy = entropy(retain_prob).mean().item()
    # forget_entropy = entropy(forget_prob).mean().item()
    # test_entropy = entropy(test_prob).mean().item()
    # val_entropy = entropy(val_prob).mean().item()
    # unseen_entropy = 0.5 * (test_entropy + val_entropy)
    # mlflow.log_metric("Retain Entropy", retain_entorpy, step=step)
    # mlflow.log_metric("Forget Entropy", forget_entropy, step=step)
    # mlflow.log_metric("Unseen Entropy", unseen_entropy, step=step)
    return X_f, Y_f, X_r, Y_r


def log_mia(
    retain_loader,
    forget_loader,
    test_loader,
    val_loader,
    model,
    step=None,
):
    """
    Perform a membership inference attack (MIA) using logistic regression and log the probability of the attack's success.
    Args:
        retain_loader (DataLoader): DataLoader for the retained dataset.
        forget_loader (DataLoader): DataLoader for the forgotten dataset.
        test_loader (DataLoader): DataLoader for the test dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        model (torch.nn.Module): The model to be attacked.
        step (int, optional): The current step or epoch number for logging purposes. Defaults to None.
    Returns:
        float: The probability of the membership inference attack's success, rounded to two decimal places.
    """
    X_f, Y_f, X_r, Y_r = get_membership_attack_data(
        retain_loader, forget_loader, test_loader, val_loader, model, step=step
    )
    clf = LogisticRegression(
        class_weight="balanced",
        solver="lbfgs",
        multi_class="multinomial",
        random_state=42,
    )
    clf.fit(X_r, Y_r)
    results = clf.predict(X_f)
    prob_mia = results.mean()
    prob_mia = round(prob_mia, 2)
    mlflow.log_metric("mia", prob_mia, step=step)
    return prob_mia
