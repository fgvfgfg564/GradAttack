import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys
import argparse
import numpy as np
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

from lib.trainer import Trainer, save_checkpoint
from lib.test import test_on_dataset
from lib.path import *
from dataset import *
from criterion import RateDistortionLoss
from nets import load_model, load_lmbda, MODELS
from adversarial_trainer import AdversarialTrainer

ROOTDIR = os.path.split(__file__)[0]

from train_baseline import add_key_args, generate_exp_name

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Simple adversarial training for LIC.")

    add_key_args(parser)

    parser.add_argument("--steps", type=int, default=1000)

    # Adversarial args
    parser.add_argument("--adv_steps", type=int, default=10)
    parser.add_argument("--adv_optimizer", type=str, choices=['Adam', 'SGD'], default='Adam')
    parser.add_argument("--adv_lr", type=float, default=1e-3)
    parser.add_argument("--adv_epsilon", type=float, default=1e-3)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "-lr",
        "--learning_rate",
        default=1e-5,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed",
        type=float,
        default=19260817,
        help="Set random seed for reproducibility",
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--continue_train", action="store_true", default=False)
    args = parser.parse_args(argv)
    return args

def generate_adv_exp_name(args):
    return (f"adversarial/{args.model}/{args.dataset}-{args.lmbda}-{args.epochs}x{args.steps_per_epoch}"
            f"-{args.steps}-{args.adv_steps}-{args.adv_optimizer}-{args.adv_lr}-{args.adv_epsilon}"
    )

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    save_path = os.path.join(ROOTDIR, generate_adv_exp_name(args))
    tb_path = os.path.join(save_path, "tensorboard/")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(tb_path)

    writer = SummaryWriter(tb_path)

    device = "cuda"

    net = load_model(args.model).to(device)
    lmbda = args.lmbda

    if args.dataset == 'vimeo90k':
        train_dataset = Vimeo90KRandom(256)
    elif args.dataset == 'liu4k':
        train_dataset = LIU4KPatches() 
    else:
        raise ValueError()
    test_dataset = Kodak(None)
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, pin_memory=True, pin_memory_device=device, num_workers=6)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, pin_memory_device=device, num_workers=1)
    
    # Test adversarial performance

    adv_trainer = AdversarialTrainer(
        params=net.parameters(),
        lr=args.learning_rate,
        optimizer_class=lambda params, lr: torch.optim.Adam(params=params, lr=lr),
        dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        criterion=RateDistortionLoss(lmbda),
        steps_per_epoch=100,
        milestones=[],
        clip_max_norm=args.clip_max_norm,
        writer=writer,
        device=device,
        save_path=save_path,
        output_interval=10,
        epsilon=args.adv_epsilon,
        adv_steps=args.adv_steps,
        adv_optimizer=args.adv_optimizer,
        adv_lr=args.adv_lr
    )
    adv_trainer.logger.info("Args: " + args.__str__())
    adv_trainer.logger.info(f"Lmbda={lmbda:.12f}")
    
    if args.checkpoint:  # load from previous checkpoint
        adv_trainer.logger.info(f"Loading {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        net.load_state_dict(checkpoint["state_dict"])
        if args.continue_train:
            adv_trainer.load_state_dict(checkpoint["trainer"])
    else:
        # Load from baseline
        baseline_path = os.path.join(ROOTDIR, generate_exp_name(args.model, args.dataset, args.lmbda, args.epochs, args.steps_per_epoch))
        ckptpath = os.path.join(baseline_path, 'last_epoch.pth.tar')
        adv_trainer.logger.info('Load weights from ' + ckptpath)
        checkpoint = torch.load(ckptpath, map_location=device)
        net.load_state_dict(checkpoint['state_dict'])

    net.eval()
    adv_trainer.test_one_epoch(net)
    performance_baseline = test_on_dataset(net, test_dataloader)

    for i in range(args.steps // 100):
        net.train()
        adv_trainer.train_one_epoch(net)
        net.eval()
        adv_trainer.test_one_epoch(net)

    performance_new = test_on_dataset(net, test_dataloader)

    dump_json(performance_baseline, os.path.join(save_path, 'baseline.json'))
    dump_json(performance_new, os.path.join(save_path, 'adversarial.json'))

    if args.save:
        save_checkpoint(
            {
                "state_dict": net.state_dict(),
                "trainer": adv_trainer.get_state_dict(),
            },
            True,
            save_path,
        )