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
    accuracy = correct / total
    accuracy = round(accuracy, 2)
    return accuracy



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
    return l2_distance

# ==============================================================================
# Bad Teaching Metrics
# https://github.com/vikram2000b/bad-teaching-unlearning/blob/main/metrics.py
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


def log_js(tmodel, gold_model, forget_dl, is_multi_label=False, step=None):
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
    js = JSDiv(model_preds, gold_model_preds)
    js = js.item()
    js = round(js, 4) 
    mlflow.log_metric("js", js)
    return js



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
    zrf = round(zrf, 4)
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
    unseen_entropy = 0.5 * (test_entropy + val_entropy)
    mlflow.log_metric("Retain Entropy", retain_entorpy, step=step)
    mlflow.log_metric("Forget Entropy", forget_entropy, step=step)
    mlflow.log_metric("Unseen Entropy", unseen_entropy, step=step)
    # diff_entropy = abs(forget_entropy - unseen_entropy)
    # mlflow.log_metric("Diff Entropy", diff_entropy, step=step)

    return X_f, Y_f, X_r, Y_r


def log_membership_attack_prob(
    retain_loader,
    forget_loader,
    test_loader,
    val_loader,
    model,
    step=None,
    is_original=False,
):
    X_f, Y_f, X_r, Y_r = get_membership_attack_data(
        retain_loader, forget_loader, test_loader, val_loader, model, step=step
    )
    # clf = SVC(C=3, gamma="auto", kernel="rbf")
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
    if step is not None:
        mlflow.log_metric("MIA_prob", prob_mia, step=step)
    else:
        if is_original:
            mlflow.log_metric("original_MIA_prob", prob_mia)
        else:
            mlflow.log_metric("MIA_prob", prob_mia)

    return prob_mia
