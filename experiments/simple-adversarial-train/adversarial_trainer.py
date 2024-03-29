import torch
import torch.nn as nn

from lib.trainer import Trainer, AverageMeter
from grad_attack import attack

class AdversarialTrainer(Trainer):
    def __init__(self, params, lr, optimizer_class, dataloader, test_dataloader, criterion, steps_per_epoch, milestones, clip_max_norm, writer, device, save_path, epsilon, adv_steps, output_interval=1000):
        super().__init__(params, lr, optimizer_class, dataloader, test_dataloader, criterion, steps_per_epoch, milestones, clip_max_norm, writer, device, save_path, output_interval)

        self.epsilon = epsilon
        self.adv_steps = adv_steps
    
    @torch.enable_grad()
    def train_one_epoch(self, forward_func):
        loss_dict = {}
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
            d_adv = attack(forward_func, d, num_steps=self.adv_steps, epsilon=self.epsilon, criterion=self.criterion)
            out_net = forward_func(d_adv)
            out_criterion = self.criterion(out_net, d_adv)
            
            for key, value in out_criterion.items():
                loss_dict.setdefault(key, AverageMeter())
                loss_dict[key].update(value.detach())

            self.optimizer.zero_grad()
            out_criterion['loss'].backward()

            if self.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], self.clip_max_norm)
            self.optimizer.step()

            if i % self.output_interval == 0:
                train_log = f"Train epoch {self.epoch}: [{i}/{self.steps_per_epoch} ({100. * i / self.steps_per_epoch:.2f}%)] "
                
                for k, v in loss_dict.items():
                    train_log += f'\t{k}: {v.avg.cpu().numpy().item():.6f} |'
                    if self.writer:
                        self.writer.add_scalar("train_"+k, v.avg.detach().cpu().numpy().item(), self.global_step)
                    v.reset()
                if self.logger is not None:
                    self.logger.info(train_log)
            
            self.global_step += 1
            
            if i == self.steps_per_epoch:
                break
        
        self.scheduler.step()
        self.epoch += 1
    
    @torch.no_grad()
    def test_adversarial(self, forward_func):
        loss_dict = {}

        for d in self.test_dataloader:
            if isinstance(d, torch.Tensor):
                d = d.to(self.device)
            else:
                d_new = []
                for each in d:
                    d_new.append(each.to(self.device))
                d = tuple(d_new)
            d_adv = attack(forward_func, d, num_steps=self.adv_steps, epsilon=self.epsilon, criterion=self.criterion)
            out_net = forward_func(d_adv)
            out_criterion = self.criterion(out_net, d_adv)
            for key, value in out_criterion.items():
                loss_dict.setdefault(key, AverageMeter())
                loss_dict[key].update(value)

        test_log = f"Testing results for epoch {self.epoch} [ADVERSARIAL]"
        for k, meter in loss_dict.items():
            test_log += f" | {k}={meter.avg:.5f}"
            if self.writer:
                self.writer.add_scalar("test_adversarial_"+k, meter.avg.cpu().numpy().item(), self.epoch)
        if self.logger:
            self.logger.info(test_log)
        loss = loss_dict["loss"].avg.cpu().numpy().item()
        is_best = loss < self.best_loss
        self.best_loss = min(loss, self.best_loss)
        
        return is_best