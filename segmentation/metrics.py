class AccuracyMetric():
    def __init__(self, global_cm):
        self.global_cm = global_cm
        self.overall_acc = 0
        self.average_acc = 0
        self.average_iou = 0

    def reset(self):
        pass
    def update_values(self):
        pass
    def get_values(self):
        return self.overall_acc, self.average_acc, self.average_iou