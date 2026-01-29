
def build_confusion_matrix():
    return {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}

def conf_mat(dice, prediction, gt, confusion_matrix, dice_thresh=0.5):
    tumor_iden = (prediction > 0).any().item()
    tumor_exists   = (gt > 0).any().item()
    if tumor_exists:
        if tumor_iden and dice > dice_thresh:
            confusion_matrix['TP'] += 1
            return 'TP'
        else:
            confusion_matrix['FN'] += 1
            return 'FN'
    else:
        if tumor_iden:
            confusion_matrix['FP'] += 1
            return 'FP'
        else:
            confusion_matrix['TN'] += 1
            return 'TN'

def accuracy(confusion_matrix, eps=1e-8):
    TP, TN, FP, FN = confusion_matrix['TP'], confusion_matrix['TN'], confusion_matrix['FP'], confusion_matrix['FN']
    return (TP + TN) / (TP + TN + FP + FN + eps)

def precision(confusion_matrix, eps=1e-8):
    TP, FP = confusion_matrix['TP'], confusion_matrix['FP']
    return TP / (TP + FP + eps)

def recall(confusion_matrix, eps=1e-8):
    TP, FN = confusion_matrix['TP'], confusion_matrix['FN']
    return TP / (TP + FN + eps)

def f1_score(confusion_matrix, eps=1e-8):
    TP, FP, FN = confusion_matrix['TP'], confusion_matrix['FP'], confusion_matrix['FN']
    return (2 * TP) / (2 * TP + FP + FN + eps)

def dice_score(prediction, gt, eps=1e-6):
    intersection = (prediction * gt).sum()  # |A cross B|
    union = prediction.sum() + gt.sum()  # |A u B|
    return (2 * intersection + eps) / (union + eps)

