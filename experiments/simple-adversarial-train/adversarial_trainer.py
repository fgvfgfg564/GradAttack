import torch
import torch.nn as nn

from lib.trainer import Trainer
from grad_attack import attack

class AdversarialTrainer(Trainer):
    def __init__(self, params, lr, optimizer_class, dataloader, test_dataloader, criterion, steps_per_epoch, milestones, clip_max_norm, writer, device, save_path, epsilon, adv_steps, output_interval=1000):
        super().__init__(params, lr, optimizer_class, dataloader, test_dataloader, criterion, steps_per_epoch, milestones, clip_max_norm, writer, device, save_path, output_interval)

        self.epsilon = epsilon
        self.adv_steps = adv_steps
    
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
            d_adv = attack(forward_func, d, self.adv_steps, epsilon=self.epsilon, criterion=self.criterion)
            out_net = forward_func(d_adv)
            out_criterion = self.criterion(out_net, d_adv)
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
