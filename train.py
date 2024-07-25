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

from network.concretenet import ConcreteNet
from dataloader.factory import DatasetFactory
from utils.losses import JointLoss
from utils.loss_tracker import LossTracker
from utils.evaluator import Evaluator

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', default='config/concretenet.yaml')
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--ckpt_path', default=None)
parser.add_argument('--random_seed', default=None)
args = parser.parse_args()

if not args.random_seed:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


class Trainer:
    def __init__(self):
        super().__init__()
        config = yaml.safe_load(open(args.config_path, 'r'))
        self.config = config

        # Create dataloaders
        train_dataset = DatasetFactory(split='train', **config['dataset'])
        self.train_dataloader = DataLoader(train_dataset, collate_fn=train_dataset.collate_fn,
                                           **config['train_dataloader'])
        val_dataset = DatasetFactory(split='val', **config['dataset'])
        self.val_dataloader = DataLoader(val_dataset, collate_fn=val_dataset.collate_fn,
                                         **config['val_dataloader'])

        # Build model, loss, optimizer and evaluator
        self.model = ConcreteNet(**config['model']).cuda()
        self.criterion = JointLoss(**config['loss']).cuda()
        self.optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                                      **config['optimizer'])
        self.loss_tracker = LossTracker()
        self.evaluator = Evaluator(**config['loss']['visual_config'])

        self.start_epoch = 1 # 1-indexing epoch count
        self.num_epochs = config['trainer']['num_epochs']
        self.save_freq = config['trainer']['save_freq']
        self.load_checkpoint(args.ckpt_path, args.resume)

    def cosine_scheduler_step(self, epoch, epsilon=1e-6):
        initial_lr = self.config['optimizer']['lr']
        warmup_epochs = self.config['scheduler']['warmup_epochs']
        lr = initial_lr if epoch < warmup_epochs else epsilon + 0.5 * (initial_lr - epsilon) * \
            (1 + math.cos(math.pi * ((epoch - warmup_epochs) / (self.num_epochs - warmup_epochs))))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def decipher_stage(self, epoch):
        epoch_thresholds = self.config['trainer']['pretraining']
        stage = 0
        if epoch > epoch_thresholds['semantic']:  stage += 1
        if epoch > epoch_thresholds['candidate']: stage += 1
        if epoch > epoch_thresholds['instance']:  stage += 1
        return stage

    def train_epoch(self, epoch, stage=None):
        stage = stage if stage is not None else self.decipher_stage(epoch)
        self.model.train()
        self.loss_tracker.reset()

        for batch in (pbar := tqdm(self.train_dataloader, total=len(self.train_dataloader))):
            self.optimizer.zero_grad()
            self.cosine_scheduler_step(epoch - 1)

            batch = {k: v.cuda() if hasattr(v, 'cuda') else v for k, v in batch.items()}
            stage = self.decipher_stage(epoch)
            ret = self.model(batch, stage=stage)
            loss_dict = self.criterion(batch, ret)
            loss_dict['loss_total'].backward()
            self.optimizer.step()

            self.loss_tracker.update(loss_dict)
            pbar.set_description(f'train:{epoch}/{self.num_epochs}({stage})' + self.loss_tracker.track())
        self.save_checkpoint(epoch)

    def eval_epoch(self, epoch):
        self.model.eval()
        self.loss_tracker.reset()
        self.evaluator.reset()
        for batch in (pbar := tqdm(self.val_dataloader, total=len(self.val_dataloader))):
            batch = {k: v.cuda() if hasattr(v, 'cuda') else v for k, v in batch.items()}
            with torch.no_grad():
                ret = self.model(batch, stage=-1)
                loss_dict = self.criterion(batch, ret)
                decode_ret = self.model.decode_output(ret)

            self.loss_tracker.update(loss_dict)
            pbar.set_description(f'eval:{epoch}' + self.loss_tracker.track())
            self.evaluator.update(batch, decode_ret)
        self.evaluator.evaluate()

    def train(self):
        for epoch in range(self.start_epoch, self.num_epochs + 1):
            stage = self.decipher_stage(epoch)
            self.train_epoch(epoch, stage)
            # if not epoch % self.save_freq:
            #     self.eval_epoch(epoch)

    def evaluate(self):
        self.eval_epoch(self.start_epoch - 1)

    def save_checkpoint(self, epoch):
        save_dir = self.config['trainer']['save_dir']
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_path = os.path.join(save_dir, f'ckpt_{epoch}.pth')
        torch.save({'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': epoch}, save_path)
        # Remove previous checkpoint if not within save frequency
        prev_path = os.path.join(save_dir, f'ckpt_{epoch - 1}.pth')
        if os.path.isfile(prev_path) and (epoch - 1) % self.save_freq:
            os.remove(prev_path)

    def load_checkpoint(self, load_path=None, resume=False):
        # If load path isn't provided, find the last epoch in save_dir
        if load_path is None and resume:
            load_dir = self.config['trainer']['save_dir']
            ckpt_epochs = [int(fn.split('_')[-1].split('.')[0]) for fn in
                           os.listdir(load_dir) if fn.endswith('.pth')]
            if not ckpt_epochs:
                return
            load_path = os.path.join(load_dir, f'ckpt_{max(ckpt_epochs)}.pth')
            assert os.path.exists(load_path)

        if load_path is None:
            return

        print(f'Loading checkpoint at {load_path}.')
        ckpt = torch.load(load_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(ckpt['model'], strict=False)
        if resume:
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.start_epoch = ckpt['epoch'] + 1


if __name__=='__main__':
    Trainer().evaluate() if args.evaluate else Trainer().train()
