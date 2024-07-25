# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os
import yaml
import argparse
import pathlib
from tqdm import tqdm
import math
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

import json

from network.factory import ModelFactory
from dataloader.factory import DatasetFactory
from utils.losses import JointLoss
from utils.loss_tracker import LossTracker
from utils.evaluator import Evaluator

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', default='config/concretenet.yaml')
parser.add_argument('--ckpt_path', default='network/checkpoints/concretenet.pth')
parser.add_argument('--random_seed', default=None)
args = parser.parse_args()

if not args.random_seed:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


class Tester:
    def __init__(self):
        super().__init__()
        config = yaml.safe_load(open(args.config_path, 'r'))
        self.config = config

        # Create dataloaders
        self.dataloaders = []
        repeat_count = 5
        self.thetas = [i*2*np.pi/repeat_count for i in range(repeat_count)]
        self.dataset = DatasetFactory(split='val', **config['dataset'])

        # Build model, loss, optimizer and evaluator
        self.model = ModelFactory(**config['model']).cuda()
        self.model.eval()
        self.load_checkpoint(args.ckpt_path)
        self.evaluator = Evaluator(**config['evaluator'])

        with open('output/sample_submission.json') as submission_file:
            self.submission = submission_file.read()

        self.iou_threshold = 0.9

    def test_epoch(self):
        for i in (pbar := tqdm(range(len(self.dataset)))):
            referred_mask_preds = []
            for j, theta in enumerate(self.thetas):
                batch = self.dataset.collate_fn([self.dataset.__getitem__(i, theta)])
                batch = {k: v.cuda() if hasattr(v, 'cuda') else v for k, v in batch.items()}

                with torch.no_grad():
                    ret = self.model(batch, stage=-1)
                    decode_ret = self.model.decode_output(ret)

                lang_output = decode_ret['lang_output'][:,decode_ret['valid_idx']]
                referred_mask_preds.append(decode_ret['mask_pred'][lang_output.argmax(1)].long())

            referred_mask_preds = torch.stack(referred_mask_preds, 1)
            final_referred_masks = []
            for referred_mask_pred in referred_mask_preds:
                # Get IoU between predictions
                intersection = (referred_mask_pred[None] * referred_mask_pred[:,None]).sum(-1)
                union = (referred_mask_pred[None] | referred_mask_pred[:,None]).sum(-1)
                iou = intersection / union
                # Find prediction with highest IoU to other predictions
                row = iou.sum(1).argmax()
                valid_idx = iou[row] > self.iou_threshold
                valid_referred_mask_pred = referred_mask_pred[valid_idx]
                num_pred = valid_referred_mask_pred.shape[0]
                final_referred_mask = valid_referred_mask_pred.sum(0) > (num_pred/2)
                final_referred_masks.append(final_referred_mask)

            decode_ret['referred_mask_pred'] = torch.stack(final_referred_masks, dim=0)

            self.evaluator.update(batch, decode_ret)
        self.evaluator.evaluate()
        self.evaluator.reset()

    def test(self):
        self.test_epoch()

    def load_checkpoint(self, load_path=None):
        print(f'Loading checkpoint at {load_path}.')
        ckpt = torch.load(load_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(ckpt['model'], strict=False)

if __name__=='__main__':
    Tester().test()

# ########################################
# class          :   IoU    AP AP@50 AP@25
# ----------------------------------------
# cabinet        :  83.0  41.8  60.8  69.5
# bed            :  94.8  59.1  82.6  86.0
# chair          :  66.0  77.0  91.5  94.3
# sofa           :  84.9  55.2  71.3  81.0
# table          :  88.8  57.6  74.9  77.9
# door           :  77.2  40.9  55.7  61.6
# window         :  70.9  32.8  48.7  61.3
# bookshelf      :  58.7  24.7  44.6  66.5
# picture        :  65.5  41.8  48.9  54.6
# counter        :  70.0  30.4  49.2  70.7
# desk           :  34.5  32.9  58.6  73.7
# curtain        :  70.7  45.0  52.9  64.1
# refrigerator   :  60.4  35.5  40.0  40.0
# shower curtain :  60.8  65.8  90.9 100.0
# toilet         :  51.3  79.9  83.3  87.5
# sink           :  75.2  53.6  78.6  90.0
# bathtub        :  86.4  67.9  83.3  83.3
# otherfurniture :  68.0  47.6  60.9  67.8
# ----------------------------------------
# average        :  70.5  49.4  65.4  73.9
# ########################################
#  U@25   U@50   M@25   M@50   O@25   O@50
# ----------------------------------------
# 86.40  82.05  42.41  38.39  50.61  46.53
# ########################################
