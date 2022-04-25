import numpy as np
from sklearn.metrics import confusion_matrix


class AccuracyMetric():

    def __init__(self, global_cm):
        self.global_cm = global_cm
        self.overall_acc = 0
        self.average_acc = 0
        self.average_iou = 0

    def reset(self):
        pass

    def update_values(self, pred_, target_, labels):
        cm = confusion_matrix(target_.ravel(), pred_.ravel(), labels=labels)
        self.global_cm += cm
        if self.global_cm.sum() > 0:
            self.overall_acc = np.trace(self.global_cm) / self.global_cm.sum()

            sums = np.sum(self.global_cm, axis=1)
            mask = (sums > 0)
            sums[sums == 0] = 1
            accuracy_per_class = np.diag(
                self.global_cm) / sums    # sum over lines
            accuracy_per_class[np.logical_not(mask)] = -1
            self.average_acc = accuracy_per_class[mask].mean()

            sums = (np.sum(self.global_cm, axis=1) +
                    np.sum(self.global_cm, axis=0) - np.diag(self.global_cm))
            mask = (sums > 0)
            sums[sums == 0] = 1
            iou_per_class = np.diag(self.global_cm) / sums
            iou_per_class[np.logical_not(mask)] = -1
            self.average_iou = iou_per_class[mask].mean()
        else:
            self.overall_acc = 0
            self.average_acc = 0
            self.average_iou = 0

    def get_values(self):
        return self.overall_acc, self.average_acc, self.average_iou