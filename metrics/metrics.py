import numpy as np
import torch

def calc_accuracy(scores, labels):
    """Returns per class accuracy and overall accuracy. Scores returned here are overall 
    scores of all the calsses, we need to select max as a label.

    Args:
        scores (torch.tensor): Predicted scores of shape (B, C, N)
        labels (torch.tensor): Groundtruth labels pf shape (B, N)
    
    Returns:
        tuple: (overall accuracy, [per class accuracies])
    """
    
    pred_labels = torch.argmax(scores, dim=1)
    num_classes = scores.shape[-2]

    a_mask = pred_labels == labels
    per_class_accuracies = []
    for c in num_classes:
        class_mask = pred_labels == c
        class_accuracy = (class_mask & a_mask).float().sum().cpu().item()
        class_accuracy /= class_mask.float().sum().cpu().item()
        per_class_accuracies.append(class_accuracy)

    return per_class_accuracies.append(a_mask.float().mean().cpu().item())

def calc_iou(scores, labels):
    """calculates intersection over union given predicted labels and ground truth labels

    Args:
        scores (torch.tensor): Predicted scores of shape (B, C, N)
        labels (torch.tensor): Groundtruth labels pf shape (B, N)

    Returns:
        tuple : (mean IOU , [per class IOU]) 
    """
    pred_labels = torch.max(scores, dim=2)
    num_classes = scores.shape[-2]
    a_mask = pred_labels == labels
    per_class_iou = []
    for c in num_classes:
        class_mask = pred_labels == c
        intr = (class_mask & a_mask).float().sum().cpu().item()
        gt_mask = labels == c
        union = (class_mask | gt_mask).float().sum().cpu().item()
        iou = intr/union
        per_class_iou.append(iou)
    return per_class_iou.append(sum(per_class_iou)/len(per_class_iou))
        
        

    