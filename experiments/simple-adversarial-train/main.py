import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys
import argparse
import numpy as np
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

from adversarial_trainer import AdversarialTrainer
from lib.trainer import save_checkpoint
from dataset import *
from criterion import RateDistortionLoss
from nets import load_model, load_lmbda

ROOTDIR = os.path.split(__file__)[0]

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Simple adversarial training for LIC.")

    # model parameters
    parser.add_argument("model", type=str)
    parser.add_argument("parameter_set", type=int)  # Always load pretrained models

    # Adversarial parameters
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--adv_steps", type=int, default=100)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
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
    parser.add_argument("--lr_epoch", nargs="+", type=int, default=[5])
    parser.add_argument("--continue_train", action="store_true", default=False)
    args = parser.parse_args(argv)
    return args

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    save_path = os.path.join(ROOTDIR, f"{args.model}-{args.parameter_set}-{args.epochs}x{args.steps_per_epoch}-{args.epsilon}-{args.adv_steps}")
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

    net = load_model(args.model, args.parameter_set)
    lmbda = load_lmbda(args.model, args.parameter_set)

    train_dataset = Vimeo90KRandom(256)
    test_dataset = Kodak(512)
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, pin_memory=True, pin_memory_device=device)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, pin_memory_device=device)

    trainer = AdversarialTrainer(
        params=net.parameters(),
        lr=args.learning_rate,
        optimizer_class=lambda params, lr: torch.optim.Adam(params=params, lr=lr),
        dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        criterion=RateDistortionLoss(lmbda),
        steps_per_epoch=args.steps_per_epoch,
        milestones=args.lr_epoch,
        clip_max_norm=args.clip_max_norm,
        writer=writer,
        device=device,
        save_path=save_path,
        epsilon=args.epsilon,
        adv_steps=args.adv_steps,
        output_interval=10,
    )
    trainer.logger.info("Args: " + args.__str__())
    
    if args.checkpoint:  # load from previous checkpoint
        trainer.logger.info(f"Loading {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        net.load_state_dict(checkpoint["state_dict"])
        if args.continue_train:
            trainer.load_state_dict(checkpoint["trainer"])

    while trainer.epoch < args.epochs:
        net.train()
        trainer.train_one_epoch(net)
        net.eval()
        is_best = trainer.test_adversarial(net)
        trainer.test_one_epoch(net)

        if args.save:
            save_checkpoint(
                {
                    "state_dict": net.state_dict(),
                    "trainer": trainer.get_state_dict(),
                },
                is_best,
                save_path,
            )
