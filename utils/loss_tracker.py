# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

from collections import defaultdict
import torch


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if torch.isnan(val):
            return
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LossTracker():
    def __init__(self):
        self.reset()

    def reset(self):
        self.tracker = defaultdict(AverageMeter)

    def update(self, loss_dict):
        for key, loss in loss_dict.items():
            self.tracker[key].update(loss.detach())

    def track(self):
        desc = ''
        for key, loss in self.tracker.items():
            desc += f' {key}:{loss.avg:.3f}'
        return desc
