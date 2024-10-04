import logging

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from tqdm import tqdm

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_accuracy_imagenet(model, dataloader, max_samples=None):
    predictions = []
    match_list = []
    model.eval()
    with torch.no_grad():
        for img_batch, labels in tqdm(dataloader):
            img_batch_gpu = img_batch.cuda()
            labels_gpu = labels.cuda()
            preds = model(img_batch_gpu)  # output (batch_size, 1000)
            digit_preds = torch.argmax(preds, dim=1)
            matches = labels_gpu == digit_preds
            predictions.append(digit_preds)
            match_list.append(matches)
            del img_batch_gpu, labels_gpu, preds, digit_preds, matches

    predictions = torch.cat(predictions)
    matches = torch.cat(match_list)
    accuracy = matches.sum() / matches.shape[0]
    del predictions, matches
    accuracy = round(accuracy.item(), 2)
    return accuracy


def log_js_imagenet(unlearned, original, forget_dl, test_dl):
    unlearned.eval()
    original.eval()
    forget_loader = torch.utils.data.DataLoader(
        forget_dl.dataset, batch_size=2048, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dl.dataset, batch_size=2048, shuffle=False
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
    mlflow.log_metric("js_div", js_div)
    return js_div


def JSDiv(p, q):
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
            model_preds.append(F.softmax(model_output, dim=1).detach().cpu())
            gold_model_preds.append(
                F.softmax(gold_model_output, dim=1).detach().cpu()
            )

    model_preds = torch.cat(model_preds, axis=0)
    gold_model_preds = torch.cat(gold_model_preds, axis=0)
    zrf = 1 - JSDiv(model_preds, gold_model_preds)
    zrf = zrf.item()
    zrf = round(zrf, 4)
    mlflow.log_metric("zrf", zrf)
    return zrf

def log_zrf_imagenet(unlearned, original, forget_dl, test_dl):
    unlearned.eval()
    original.eval()
    unlearned_preds = []
    original_preds = []
    with torch.no_grad():
        for x, y in forget_dl.dataset:
            x = x.unsqueeze(0).to(DEVICE)
            unlearned_preds.append(F.softmax(unlearned(x), dim=1).detach().cpu())

            count = 0
            original_samples = []
            for xt, yt in test_dl.dataset:
                if yt == y:
                    count += 1
                    original_samples.append(F.softmax(original(xt), dim=1).detach().cpu())
                    if count == 30:
                        original_preds.append(original_preds.mean(axis=0))
                        break

    unlearned_preds = torch.cat(unlearned_preds, axis=0)
    original_preds = torch.cat(original_preds, axis=0)
    zrf = 1 - JSDiv(unlearned_preds, original_preds)
    zrf = zrf.item()
    zrf = round(zrf, 4)
    mlflow.log_metric("ZRF", zrf)
    return zrf


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
    # clf = SVC(C=3, gamma="auto", kernel="rbf", random_state=42)
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
