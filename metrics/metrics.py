import numpy as np
import torch

def calc_accuracy(scores, labels):
    """Returns per class accuracy and overall accuracy. Scores returned here are overall 
    scores of all the calsses, we need to select max as a label.

    Args:
        scores (torch.tensor): Predicted scores of shape (B, C, N)
        labels (torch.tensor): Groundtruth labels pf shape (B, N)
    
    Returns:
        tuple: ([per class accuracies, overall_accuracy])
    """
    # torch.save(labels, 'labels.pt')
    # torch.save(scores, 'scores.pt')    
    pred_labels = torch.argmax(scores.detach(), dim=1)
    num_classes = scores.shape[-2]

    labels = labels.detach()
    a_mask = pred_labels == labels
    per_class_accuracies = []
    for c in range(num_classes):
        class_mask = pred_labels == c
        class_accuracy = (class_mask & a_mask).float().sum().cpu().item()
        class_accuracy /= (class_mask.float().sum().cpu().item() + 1e-05)
        per_class_accuracies.append(class_accuracy)

    per_class_accuracies.append(a_mask.float().mean().cpu().item())
    return per_class_accuracies

def calc_iou(scores, labels):
    """calculates intersection over union given predicted labels and ground truth labels

    Args:
        scores (torch.tensor): Predicted scores of shape (B, C, N)
        labels (torch.tensor): Groundtruth labels pf shape (B, N)

    Returns:
        tuple : ([per class IOU, mean_IOU]) 
    """
    pred_labels = torch.argmax(scores.detach(), dim=1)
    num_classes = scores.shape[-2]

    labels = labels.detach()
    a_mask = pred_labels == labels
    per_class_iou = []
    for c in range(num_classes):
        class_mask = pred_labels == c
        intr = (class_mask & a_mask).float().sum().cpu().item()
        gt_mask = labels == c
        union = (class_mask | gt_mask).float().sum().cpu().item()
        iou = intr/(union + 1e-05)
        per_class_iou.append(iou)
    per_class_iou.append(sum(per_class_iou)/len(per_class_iou))
    return per_class_iou
        
import numpy as np
import torch

def accuracy(scores, labels):
    r"""
        Compute the per-class accuracies and the overall accuracy # TODO: complete doc
        Parameters
        ----------
        scores: torch.FloatTensor, shape (B?, C, N)
            raw scores for each class
        labels: torch.LongTensor, shape (B?, N)
            ground truth labels
        Returns
        -------
        list of floats of length num_classes+1 (last item is overall accuracy)
    """
    num_classes = scores.size(-2) # we use -2 instead of 1 to enable arbitrary batch dimensions

    predictions = torch.max(scores, dim=-2).indices

    accuracies = []

    accuracy_mask = predictions == labels
    for label in range(num_classes):
        label_mask = labels == label
        per_class_accuracy = (accuracy_mask & label_mask).float().sum()
        per_class_accuracy /= label_mask.float().sum()
        accuracies.append(per_class_accuracy.cpu().item())
    # overall accuracy
    accuracies.append(accuracy_mask.float().mean().cpu().item())
    return accuracies

def intersection_over_union(scores, labels):
    r"""
        Compute the per-class IoU and the mean IoU # TODO: complete doc
        Parameters
        ----------
        scores: torch.FloatTensor, shape (B?, C, N)
            raw scores for each class
        labels: torch.LongTensor, shape (B?, N)
            ground truth labels
        Returns
        -------
        list of floats of length num_classes+1 (last item is mIoU)
    """
    num_classes = scores.size(-2) # we use -2 instead of 1 to enable arbitrary batch dimensions

    predictions = torch.max(scores, dim=-2).indices

    ious = []

    for label in range(num_classes):
        pred_mask = predictions == label
        labels_mask = labels == label
        iou = (pred_mask & labels_mask).float().sum() / (pred_mask | labels_mask).float().sum()
        ious.append(iou.cpu().item())
    ious.append(np.nanmean(ious))
    return ious

    