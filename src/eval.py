import numpy as np
import torch
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_accuracy(model, dataloader):
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
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = 100 * correct / total
    accuracy = round(accuracy, 2)
    return accuracy


def mia(model, tr_loader, te_loader, threshold, n_classes=10):
    """
    Computes the membership inference attack (MIA) metrics based on a given threshold.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        tr_loader (torch.utils.data.DataLoader): The data loader for the training set.
        te_loader (torch.utils.data.DataLoader): The data loader for the test set.
        threshold (float): The threshold value to use for the MIA.
        n_classes (int, optional): The number of classes in the dataset (default: 10).

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
        criterion = torch.nn.CrossEntropyLoss(reduction="none").to(DEVICE)
        tp = torch.zeros(n_classes, device=DEVICE)
        fp = torch.zeros(n_classes, device=DEVICE)
        tn = torch.zeros(n_classes, device=DEVICE)
        fn = torch.zeros(n_classes, device=DEVICE)

        # on training loader (members, i.e., positive class)
        for _, (inputs, labels) in enumerate(tr_loader):
            inputs = inputs.to(device=DEVICE, non_blocking=True)
            labels = labels.to(device=DEVICE, non_blocking=True)

            outputs = model(inputs)
            losses = criterion(outputs, labels)
            # with global threshold
            predictions = losses < threshold
            # class-wise confusion matrix values
            for i in range(n_classes):
                preds = predictions[labels == i]
                n_member_pred = preds.sum()
                tp[i] += n_member_pred
                fn[i] += len(preds) - n_member_pred

        # on test loader (non-members, i.e., negative class)
        for _, (inputs, labels) in enumerate(te_loader):
            inputs = inputs.to(device=DEVICE, non_blocking=True)
            labels = labels.to(device=DEVICE, non_blocking=True)

            outputs = model(inputs)
            losses = criterion(outputs, labels)
            # with global threshold
            predictions = losses < threshold
            # class-wise confusion matrix values
            for i in range(n_classes):
                preds = predictions[labels == i]
                n_member_pred = preds.sum()
                fp[i] += n_member_pred
                tn[i] += len(preds) - n_member_pred

        # # class-wise bacc, tpr, fpr computations
        # class_tpr, class_fpr = torch.zeros(n_classes, device=DEVICE),  torch.zeros(n_classes, device=DEVICE)
        # class_bacc = torch.zeros(n_classes, device=DEVICE)
        # for i in range(n_classes):
        #     class_i_tpr, class_i_tnr = tp[i]/(tp[i] + fn[i]), tn[i]/(tn[i] + fp[i])
        #     class_tpr[i], class_fpr[i] = class_i_tpr, 1-class_i_tnr
        #     class_bacc[i] = (class_i_tpr+class_i_tnr)/2

        # dataset-wise bacc, tpr, fpr computations
        ds_tp, ds_fp = tp.sum(), fp.sum()
        ds_tn, ds_fn = tn.sum(), fn.sum()
        ds_tpr, ds_tnr = ds_tp / (ds_tp + ds_fn), ds_tn / (ds_tn + ds_fp)
        ds_bacc, ds_fpr = (ds_tpr + ds_tnr) / 2, 1 - ds_tnr

        ds_bacc = round(ds_bacc.item() * 100, 2)
        ds_tpr = round(ds_tpr.item() * 100, 2)
        ds_fpr = round(ds_fpr.item() * 100, 2)

    return (ds_bacc, ds_tpr, ds_fpr, ds_tp, ds_fn)


def get_forgetting_rate(bt, bf, af):
    """
    Computes the forgetting rate (FR) of an unlearned or retrained model

    Args:
        bt (int): True Negative of samples' membership before unlearning or retraining (i.e., of the original model)
        bf (int): False Negative of samples' membership before unlearning or retraining (i.e., of the original model)
        af (int): False Negative of samples' membership after unlearning or retraining (i.e., of the unlearned or retrained model)

    Returns:
        float: The forgetting rate of the model, as a percentage.
    """

    fr = (af - bf) / bt
    fr = round(fr.item() * 100, 2)
    return fr


def get_js_div(ref_model, eval_model, forget_loader):
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
        for inputs, _ in forget_loader:
            inputs = inputs.to(DEVICE)
            ref_probs = F.softmax(ref_model(inputs))
            eval_probs = F.softmax(eval_model(inputs))
            ref_probs_list.append(ref_probs.detach().cpu())
            eval_probs_list.append(eval_probs.detach().cpu())
    ref_probs_2d = torch.cat(ref_probs_list, axis=0)
    eval_probs_2d = torch.cat(eval_probs_list, axis=0)
    ref_log_probs = torch.log(ref_probs_2d)
    eval_log_probs = torch.log(eval_probs_2d)

    # JS divergence computation
    m = 0.5 * (ref_probs_2d + eval_probs_2d)
    js_div = 0.5 * (F.kl_div(ref_log_probs, m) + F.kl_div(eval_log_probs, m))
    js_div = round(js_div.item(), 4)
    return js_div


def get_l2_weight_distance(ref_model, eval_model):
    """
    Compute the L2 weight distance between two models.
    L2 weight distance is a measure of the Euclidean distance between the weights of two models.

    Args:
        ref_model (torch.nn.Module): The reference model to compare with.
        eval_model (torch.nn.Module): The model to evaluate.

    Returns:
        float: The computed L2 weight distance.
    """

    ref_weights = np.concatenate(
        [p.detach().cpu().numpy().flatten() for p in ref_model.parameters()]
    )
    eval_weights = np.concatenate(
        [p.detach().cpu().numpy().flatten() for p in eval_model.parameters()]
    )
    l2_distance = np.linalg.norm(ref_weights - eval_weights, ord=2)
    l2_distance = round(float(l2_distance), 2)
    return l2_distance


def functional_unlearning_percentage(
    mia_tp,
    mia_tp_original,
    acc_retain,
    acc_retain_original,
    acc_test,
    acc_test_original,
):
    """
    Our metric for functional unlearning percentage (FUP).
    Calculates the functional unlearning percentage based on the true positives, false negatives, accuracy of the retained dataset, accuracy of the original retained dataset, accuracy of the test dataset, and accuracy of the original test dataset.

    Args:
    mia_tp (int): Number of true positives.
    mia_tp_original (int): Number of false negatives.
    acc_retain (float): Model's accuracy on the retained dataset.
    acc_retain_original (float): Original model's accuracy on the retained dataset.
    acc_test (float): Model's accuracy on the test dataset.
    acc_test_original (float): Original model's accurach on the test dataset.

    Returns:
    float: The functional unlearning percentage.
    """
    fup = (
        (1 - (mia_tp / mia_tp_original))
        * (acc_retain / acc_retain_original)
        * (acc_test / acc_test_original)
    )
    if isinstance(fup, torch.Tensor):
        fup = round(fup.item() * 100, 2)
    else:
        fup = round(fup * 100, 2)
    return fup
