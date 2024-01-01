import torch
import torch.nn as nn
import os
import sys
import argparse
import numpy as np
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

from adversarial_trainer import AdversarialTrainer
from dataset import *
from criterion import RateDistortionLoss
from nets import load_model

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Simple adversarial training for LIC.")

    # model parameters
    parser.add_argument("--model", type=str)
    parser.add_argument("")

    parser.add_argument(
        "-e",
        "--epochs",
        default=10,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "--steps_per_epoch",
        default=100,
        type=int,
        help="Steps per epoch (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        default=1e-4,
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
        default=-1.,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument(
        "--type", type=str, default="mse", help="loss type", choices=["mse", "ms-ssim"]
    )
    parser.add_argument("--save_path", type=str, default="./debug", help="save_path")
    parser.add_argument("--lr_epoch", nargs="+", type=int, default=[5])
    parser.add_argument("--continue_train", action="store_true", default=False)
    args = parser.parse_args(argv)
    return args

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    save_path = os.path.join(args.save_path)
    tb_path = os.path.join(save_path, "tensorboard/")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(tb_path)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    writer = SummaryWriter(tb_path)

    device = "cuda"

    net = 

    trainer = Trainer(
        params=net.learnable_parameters(),
        lr=args.learning_rate,
        optimizer_class=lambda params, lr: torch.optim.Adam(params=params, lr=lr),
        dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        criterion=DistortionLoss(args.type),
        steps_per_epoch=args.steps_per_epoch,
        milestones=args.lr_epoch,
        clip_max_norm=args.clip_max_norm,
        writer=writer,
        device=device,
        save_path=save_path,
        output_interval=50,
    )
    trainer.logger.info("Args: " + args.__str__())
    
    if args.checkpoint:  # load from previous checkpoint
        trainer.logger.info(f"Loading {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        net.load_state_dict(checkpoint["state_dict"])
        if args.continue_train:
            trainer.load_state_dict(checkpoint["trainer"])

    while trainer.epoch < args.epochs:
        net.inr.train()
        trainer.train_one_epoch(net)
        net.inr.eval()
        is_best = trainer.test_one_epoch(net)

        if args.save:
            save_checkpoint(
                {
                    "state_dict": net.state_dict(),
                    "trainer": trainer.get_state_dict(),
                },
                is_best,
                save_path,
            )
