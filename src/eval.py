import torch

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
    """
    model.eval()
    with torch.inference_mode():
        criterion = torch.nn.CrossEntropyLoss(reduction="none").to(DEVICE)
        tp, fp = torch.zeros(n_classes, device=DEVICE), torch.zeros(
            n_classes, device=DEVICE
        )
        tn = torch.zeros(n_classes).to(DEVICE),
        fn = torch.zeros(n_classes, device=DEVICE)

        # on training loader (members, i.e., positive class)
        for _, (inputs, labels) in enumerate(tr_loader):
            inputs = inputs.to(device=DEVICE, non_blocking=True)
            labels.to(device=DEVICE, non_blocking=True)

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
            labels.to(device=DEVICE, non_blocking=True)
            
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

    return (
        ds_bacc * 100,
        ds_tpr * 100,
        ds_fpr * 100,
    )  # , (class_bacc, class_tpr, class_fpr)
