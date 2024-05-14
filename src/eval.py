import logging

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_accuracy(model, dataloader, is_multi_label=False):
    """
    Computes the accuracy of a PyTorch model on a given dataset.

    Args:
        model (torch.nn.Module): The PyTorch model to evaluate.
        dataloader (torch.utils.data.DataLoader): The PyTorch dataloader for the dataset.

    Returns:
        float: The accuracy of the model on the dataset, as a percentage.
    """
    correct = 0
    total = 0
    model.to(DEVICE)
    with torch.inference_mode():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            if is_multi_label == False:
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
            else:
                # multi-label classification
                probs = torch.sigmoid(outputs)
                predicted = (probs > 0.5).int()
                total += targets.size(0) * targets.size(1)
                correct += (predicted == targets.int()).sum().item()
    accuracy = 100 * correct / total
    accuracy = round(accuracy, 2)
    return accuracy


def jensen_shannon_divergence(p, q):
    """
    Compute Jensen-Shannon Divergence between two probability distributions.
    """
    # Convert probabilities to numpy arrays
    p = np.array(p)
    q = np.array(q)

    # Compute the average probability distribution
    m = 0.5 * (p + q)
    if np.isnan(m.any()):
        q = np.array(q) + 1e-32  # to avoid log(0)
        m = 0.5 * (p + q)

    # Compute Jensen-Shannon Divergence
    jsd = 0.5 * (np.sum(p * np.log2(p / m)) + np.sum(q * np.log2(q / m)))

    return jsd


def log_js_div(ref_model, eval_model, train_loader, dataset):
    """
    Compute the JS divergence of the outputs of two models.
    JS divergence is a measure of the dissimilarity between two probability distributions.

    Args:
        ref_model (torch.nn.Module): The reference model to compare with.
        eval_model (torch.nn.Module): The model to evaluate.
        forget_loader (torch.utils.data.DataLoader): The data loader for the forget set.

    Returns:
        float: The computed JS divergence.
    """

    ref_model.to(DEVICE)
    eval_model.to(DEVICE)
    ref_probs_list = []
    eval_probs_list = []
    with torch.inference_mode():
        for inputs, _ in train_loader:
            inputs = inputs.to(DEVICE)
            if dataset != "mucac":
                ref_probs = F.softmax(ref_model(inputs))
                eval_probs = F.softmax(eval_model(inputs))
            else:
                ref_probs = torch.sigmoid(ref_model(inputs))
                eval_probs = torch.sigmoid(eval_model(inputs))

            ref_probs_list.append(ref_probs.detach().cpu())
            eval_probs_list.append(eval_probs.detach().cpu())

    ref_probs_list = torch.cat(ref_probs_list, dim=0)
    eval_probs_list = torch.cat(eval_probs_list, dim=0)

    jsd_values = [
        jensen_shannon_divergence(p1, p2)
        for p1, p2 in zip(ref_probs_list, eval_probs_list)
    ]
    avg_jsd = np.mean(jsd_values)
    avg_jsd = round(avg_jsd.item(), 4)
    mlflow.log_metric("js_div", avg_jsd)


def log_l2_params_distance(ref_model, eval_model):
    """
    Compute the L2 weight distance between two models.
    L2 weight distance is a measure of the Euclidean distance between the params of two models.
    It has been cross-checked that this function has the same functionality as the distance()
    function in the SelctiveForgetting(Fisher) repo (https://github.com/AdityaGolatkar/SelectiveForgetting)

    Args:
        ref_model (torch.nn.Module): The reference model to compare with.
        eval_model (torch.nn.Module): The model to evaluate.

    Returns:
        float: The computed L2 weight distance.
    """

    ref_params = np.concatenate(
        [p.detach().cpu().numpy().flatten() for p in ref_model.parameters()]
    )
    eval_params = np.concatenate(
        [p.detach().cpu().numpy().flatten() for p in eval_model.parameters()]
    )
    l2_distance = np.linalg.norm(ref_params - eval_params, ord=2)
    l2_distance = round(float(l2_distance), 2)
    mlflow.log_metric("l2_distance", l2_distance)


# def distance(model, model0):
#     """ https://github.com/AdityaGolatkar/SelectiveForgetting """"
#     distance = 0
#     normalization = 0
#     for (k, p), (k0, p0) in zip(model.named_parameters(), model0.named_parameters()):
#         # space='  ' if 'bias' in k else ''
#         current_dist = (p.data - p0.data).pow(2).sum().item()
#         current_norm = p.data.pow(2).sum().item()
#         distance += current_dist
#         normalization += current_norm
#     print(f"Distance: {np.sqrt(distance)}")
#     print(f"Normalized Distance: {1.0*np.sqrt(distance/normalization)}")
#     return 1.0 * np.sqrt(distance / normalization)

# ==============================================================================
# Bad Teaching Metrics: https://github.com/vikram2000b/bad-teaching-unlearning/blob/main/metrics.py
# ==============================================================================

# ===========================
# ZRF (JSDiv)
# ===========================


def JSDiv(p, q):
    m = (p + q) / 2
    # This check doesn't exist in the original code
    if np.isnan(m.any()):
        q = q + 1e-32  # to avoid log(0)
        m = (p + q) / 2
    return 0.5 * F.kl_div(torch.log(p), m) + 0.5 * F.kl_div(torch.log(q), m)


# ZRF/UnLearningScore
def log_zrf(tmodel, gold_model, forget_dl, is_multi_label=False, step=None):
    model_preds = []
    gold_model_preds = []
    with torch.no_grad():
        for x, _ in forget_dl.dataset:
            x = x.unsqueeze(0).to(DEVICE)
            model_output = tmodel(x)
            gold_model_output = gold_model(x)
            if is_multi_label == False:
                model_preds.append(F.softmax(model_output, dim=1).detach().cpu())
                gold_model_preds.append(
                    F.softmax(gold_model_output, dim=1).detach().cpu()
                )
            else:
                model_preds.append(torch.sigmoid(model_output).detach().cpu())
                gold_model_preds.append(torch.sigmoid(gold_model_output).detach().cpu())

    model_preds = torch.cat(model_preds, axis=0)
    gold_model_preds = torch.cat(gold_model_preds, axis=0)
    zrf = 1 - JSDiv(model_preds, gold_model_preds)
    zrf = zrf.item()
    zrf = round(zrf, 4) * 100
    mlflow.log_metric("ZRF", zrf)


# ===========================
# MIA bad-teaching and ssd
# ===========================


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
            data, _ = batch
            output = model(data)
            prob.append(F.softmax(output, dim=-1).data)
    return torch.cat(prob)


def get_membership_attack_data(
    retain_loader, forget_loader, test_loader, val_loader, model, step=None
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
    Y_f = np.concatenate([np.ones(len(forget_prob))])

    # Log Mean Entropy
    retain_entorpy = entropy(retain_prob).mean().item()
    forget_entropy = entropy(forget_prob).mean().item()
    test_entropy = entropy(test_prob).mean().item()
    val_entropy = entropy(val_prob).mean().item()
    mlflow.log_metric("Retain Entropy", retain_entorpy, step=step)
    mlflow.log_metric("Forget Entropy", forget_entropy, step=step)
    mlflow.log_metric("Unseen Entropy", 0.5 * (test_entropy + val_entropy), step=step)

    return X_f, Y_f, X_r, Y_r


def log_membership_attack_prob(
    retain_loader, forget_loader, test_loader, val_loader, model, step=None
):
    X_f, Y_f, X_r, Y_r = get_membership_attack_data(
        retain_loader, forget_loader, test_loader, val_loader, model, step=step
    )
    # clf = SVC(C=3, gamma="auto", kernel="rbf")
    clf = LogisticRegression(
        class_weight="balanced", solver="lbfgs", multi_class="multinomial"
    )
    clf.fit(X_r, Y_r)
    results = clf.predict(X_f)
    prob_mia = results.mean() * 100
    prob_mia = round(prob_mia, 2)
    if step is not None:
        mlflow.log_metric("MIA_prob", prob_mia, step=step)
    else:
        mlflow.log_metric("MIA_prob", prob_mia)
