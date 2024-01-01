import torch
import torch.nn as nn
import logging
import os

import numpy as np

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Trainer:
    def __init__(self, params, lr, optimizer_class, dataloader, test_dataloader, criterion, steps_per_epoch, milestones, clip_max_norm, writer, device, save_path, output_interval=1000):
        self.optimizer = optimizer_class(params, lr=lr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.2)
        self.dataloader = dataloader
        self.test_dataloader = test_dataloader
        self.criterion = criterion
        self.steps_per_epoch = steps_per_epoch
        self.clip_max_norm = clip_max_norm
        self.writer = writer
        self.save_path = save_path
        self.device = device
        self.output_interval = output_interval

        self.global_step = 0
        self.epoch = 0
        self.best_loss = float("inf")

        self.config_logger()
    
    def train_one_epoch(self, forward_func):
        if self.logger is not None:
            self.logger.info(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
        for i, d in enumerate(self.dataloader):
            if isinstance(d, torch.Tensor):
                d = d.to(self.device)
            else:
                d_new = []
                for each in d:
                    d_new.append(each.to(self.device))
                d = tuple(d_new)
            out_net = forward_func(d)
            out_criterion = self.criterion(out_net, d)
            self.optimizer.zero_grad()
            out_criterion['loss'].backward()

            if self.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], self.clip_max_norm)
            self.optimizer.step()

            if i % self.output_interval == 0:
                train_log = f"Train epoch {self.epoch}: [{i}/{self.steps_per_epoch} ({100. * i / self.steps_per_epoch:.2f}%)] "
                
                for k, v in out_criterion.items():
                    train_log += f'\t{k}: {v.item():.6f} |'
                    if self.writer:
                        self.writer.add_scalar(k, v.detach().cpu().numpy().item(), self.global_step)
                if self.logger is not None:
                    self.logger.info(train_log)
            
            self.global_step += 1
            
            if i == self.steps_per_epoch:
                break
        
        self.scheduler.step()
        self.epoch += 1

    def config_logger(self):
        logger = logging.Logger("File logger", level=logging.DEBUG)

        formatter = logging.Formatter("[%(asctime)s | %(levelname)s | %(funcName)s] %(message)s")

        handler_info = logging.FileHandler(os.path.join(self.save_path, "train.log"))
        handler_info.setLevel(logging.INFO)
        handler_info.setFormatter(formatter)
        logger.addHandler(handler_info)
        handler_std = logging.StreamHandler()
        handler_std.setLevel(logging.INFO)
        handler_std.setFormatter(formatter)
        logger.addHandler(handler_std)

        self.logger = logger
    
    def test_one_epoch(self, forward_func):
        with torch.no_grad():
            losses = []
            loss_dict = {}

            for d in self.test_dataloader:
                if isinstance(d, torch.Tensor):
                    d = d.to(self.device)
                else:
                    d_new = []
                    for each in d:
                        d_new.append(each.to(self.device))
                    d = tuple(d_new)
                out_net = forward_func(d)
                out_criterion = self.criterion(out_net, d)
                for key, value in out_criterion.items():
                    loss_dict.setdefault(key, AverageMeter())
                    loss_dict[key].update(value)

            test_log = f"Testing results for epoch {self.epoch}"
            loss_items = []
            for k, meter in loss_dict.items():
                test_log += f" | {k}={meter.avg:.5f}"
                if self.writer:
                    self.writer.add_scalar(k, meter.avg.cpu().numpy().item(), self.epoch)
            if self.logger:
                self.logger.info(test_log)
            loss = loss_dict["loss"].avg.cpu().numpy().item()

            self.writer.add_scalar("test_loss", loss, self.epoch)
            is_best = loss < self.best_loss
            self.best_loss = min(loss, self.best_loss)
        
        return is_best

    def get_state_dict(self):
        return {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_loss": self.best_loss,
        }
    
    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])
        self.epoch = state_dict['epoch']
        self.global_step = state_dict['global_step']
        self.best_loss = state_dict['best_loss']