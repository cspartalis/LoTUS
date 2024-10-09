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
