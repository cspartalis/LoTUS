import numpy as np
import torch
import torch.nn.functional as F

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


def mia(model, tr_loader, val_loader, threshold, is_multi_label=False):
    """
    Computes the membership inference attack (MIA) metrics based on a given threshold.
    https://github.com/TinfoilHat0/MemberInference-by-LossThreshold/blob/main/src/my_utils.py#L43
    The code is slightly modified, no class-wise metrics are computed, only dataset-wise metrics.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        tr_loader (torch.utils.data.DataLoader): The data loader for the training set.
        val_loader (torch.utils.data.DataLoader): The data loader for the test set.
        threshold (float): The threshold value to use for the MIA.
        is_multi_label (bool, optional): If True, the dataset is multi-label. Defaults to False.

    Returns:
        A tuple of two tuples, containing the dataset-wise and class-wise MIA metrics, respectively.
        Each tuple contains three float tensors with the following metrics:
        - Balanced accuracy (bacc): the average of true positive rate (tpr) and true negative rate (tnr).
        - True positive rate (tpr): the proportion of actual members that are correctly identified as such.
        - False positive rate (fpr): the proportion of non-members that are incorrectly identified as members.
        The first tensor in each tuple contains the dataset-wise metrics, while the other tensors contain the
        metrics for each class in the dataset.
        - True positives (tp): the number of actual members that are correctly identified as such.
        - False negatives (fn): the number of actual members that are incorrectly identified as non-members.
    """
    model.to(DEVICE)
    model.eval()
    with torch.inference_mode():
        if is_multi_label == False:
            criterion = torch.nn.CrossEntropyLoss(reduction="none").to(DEVICE)
        else:
            criterion = torch.nn.BCEWithLogitsLoss(reduction="none").to(DEVICE)
        tp, fp, tn, fn = 0, 0, 0, 0

        # on training loader (members, i.e., positive class)
        for _, (inputs, labels) in enumerate(tr_loader):
            inputs = inputs.to(device=DEVICE, non_blocking=True)
            labels = labels.to(device=DEVICE, non_blocking=True)

            outputs = model(inputs)
            losses = criterion(outputs, labels)
            if is_multi_label:
                losses = losses.mean(axis=1)
            # with global threshold
            predictions = losses < threshold
            tp += predictions.sum().item()
            fn += (~predictions).sum().item()

        # on val loader (non-members, i.e., negative class)
        for _, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.to(device=DEVICE, non_blocking=True)
            labels = labels.to(device=DEVICE, non_blocking=True)

            outputs = model(inputs)
            losses = criterion(outputs, labels)
            if is_multi_label:
                losses = losses.mean(axis=1)
            # with global threshold
            predictions = losses < threshold
            fp += predictions.sum().item()
            tn += (~predictions).sum().item()

        # dataset-wise bacc, tpr, fpr computations
        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)
        bacc = (tpr + tnr) / 2

        bacc = round(bacc * 100, 2)
        tpr = round(tpr * 100, 2)
        tnr = round(tnr * 100, 2)

    return (bacc, tpr, tnr, tp, fn)


def get_forgetting_rate(bt, bf, af):
    """
    Computes the forgetting rate (FR) of an unlearned or retrained model

    Args:
        bt (int): Before true positives  (forget samples were identified as members)
        bf (int): Before false negatives (forget samples were not identified as members)
        af (int): After false negatives  (forget samples were not identified as members)
    Returns:
        float: The forgetting rate of the model, as a percentage.
    """

    fr = (af - bf) / bt
    fr = round(fr * 100, 2)
    return fr


def jensen_shannon_divergence(p, q):
    """
    Compute Jensen-Shannon Divergence between two probability distributions.
    """
    # Convert probabilities to numpy arrays
    p = np.array(p)
    q = np.array(q)

    # Compute the average probability distribution
    m = 0.5 * (p + q)

    # Compute Jensen-Shannon Divergence
    jsd = 0.5 * (np.sum(p * np.log2(p / m)) + np.sum(q * np.log2(q / m)))

    return jsd


def get_js_div(ref_model, eval_model, train_loader, dataset):
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
    return avg_jsd


def get_l2_params_distance(ref_model, eval_model):
    """
    Compute the L2 weight distance between two models.
    L2 weight distance is a measure of the Euclidean distance between the params of two models.
    It has been cross-checked that this function has the same functionality as the distance()
    function in the SelctiveForgetting(Fisher) repo (https://github.com/AdityaGolatkar/SelectiveForgetting)
    @TODO The distance() also provides a normalized l2 distance

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
    l2_distance_norm = l2_distance / np.linalg.norm(ref_params, ord=2)
    l2_distance = round(float(l2_distance), 2)
    l2_distance_norm = round(float(l2_distance_norm), 2)
    return l2_distance, l2_distance_norm


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
