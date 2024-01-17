import torch
import torch.nn as nn

from lib.trainer import Trainer, AverageMeter
from grad_attack import attack

class AdversarialTrainer(Trainer):
    def __init__(self, params, lr, optimizer_class, dataloader, test_dataloader, criterion, steps_per_epoch, milestones, clip_max_norm, writer, device, save_path, epsilon, adv_steps, adv_optimizer, adv_lr, output_interval=1):
        super().__init__(params, lr, optimizer_class, dataloader, test_dataloader, criterion, steps_per_epoch, milestones, clip_max_norm, writer, device, save_path, output_interval)

        self.epsilon = epsilon
        self.adv_steps = adv_steps
        if adv_optimizer == 'Adam':
            self.adv_optimizer = torch.optim.Adam
        elif adv_optimizer == 'SGD':
            self.adv_optimizer = torch.optim.SGD
        else:
            raise ValueError(f"Invalid optimizer: {adv_optimizer}")
        self.adv_lr = adv_lr
        self.enabled = True
    
    def enable(self):
        self.enabled = True
    
    def disable(self):
        self.enabled = False
    
    def forward_hook(self, forward_func, x):
        if self.enabled:
            x = attack(forward_func, x, num_steps=self.adv_steps, epsilon=self.epsilon, criterion=self.criterion, init_lr=self.adv_lr, optimizer_class=self.adv_optimizer, show_info=True)
        return x
    
    # @torch.no_grad()
    # def test_adversarial(self, forward_func):
    #     loss_dict = {}

    #     for d in self.test_dataloader:
    #         if isinstance(d, torch.Tensor):
    #             d = d.to(self.device)
    #         else:
    #             d_new = []
    #             for each in d:
    #                 d_new.append(each.to(self.device))
    #             d = tuple(d_new)
    #         d_adv = attack(forward_func, d, num_steps=self.adv_steps, epsilon=self.epsilon, criterion=self.criterion)
    #         out_net = forward_func(d_adv)
    #         out_criterion = self.criterion(out_net, d_adv)
    #         for key, value in out_criterion.items():
    #             loss_dict.setdefault(key, AverageMeter())
    #             loss_dict[key].update(value)

    #     test_log = f"Testing results for epoch {self.epoch} [ADVERSARIAL]"
    #     for k, meter in loss_dict.items():
    #         test_log += f" | {k}={meter.avg:.5f}"
    #         if self.writer:
    #             self.writer.add_scalar("test_adversarial_"+k, meter.avg.cpu().numpy().item(), self.epoch)
    #     if self.logger:
    #         self.logger.info(test_log)
    #     loss = loss_dict["loss"].avg.cpu().numpy().item()
    #     is_best = loss < self.best_loss
    #     self.best_loss = min(loss, self.best_loss)
        
    #     return is_best