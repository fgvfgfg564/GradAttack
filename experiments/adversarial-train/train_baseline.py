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
from dataset import *
from criterion import RateDistortionLoss
from nets import load_model, load_lmbda, MODELS

ROOTDIR = os.path.split(__file__)[0]


def add_key_args(parser):
    """
    Args that will vary among experiments
    """
    # model parameters
    parser.add_argument("model", type=str, choices=list(MODELS.keys()))

    parser.add_argument("--lmbda", type=float, default=None)
    parser.add_argument("--dataset", type=str, choices=["vimeo90k", "liu4k"])
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    ## TCM is trained with batch_size=4
    parser.add_argument(
        "--steps_per_epoch",
        default=10000,
        type=int,
        help="Steps per epoch (default: %(default)s)",
    )
    parser.add_argument(
        "--type", type=str, default="mse", help="loss type", choices=["mse", "ms-ssim"]
    )


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Simple adversarial training for LIC.")

    add_key_args(parser)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "-lr",
        "--learning_rate",
        default=5e-5,
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
    parser.add_argument("--lr_epoch", nargs="+", type=int, default=[40])
    parser.add_argument("--continue_train", action="store_true", default=False)
    args = parser.parse_args(argv)
    return args


def generate_exp_name(args):
    return f"baseline/{args.model}-{args.dataset}-{args.lmbda}-{args.epochs}x{args.steps_per_epoch}"


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    save_path = os.path.join(ROOTDIR, generate_exp_name(args))
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

    net = load_model(args.model).to(device)
    lmbda = args.lmbda

    if args.dataset == "vimeo90k":
        train_dataset = Vimeo90KRandom(256)
    elif args.dataset == "liu4k":
        train_dataset = LIU4KPatches()
    else:
        raise ValueError()
    test_dataset = Kodak(512)
    train_dataloader = DataLoader(
        train_dataset,
        args.batch_size,
        shuffle=True,
        pin_memory=True,
        pin_memory_device=device,
        num_workers=6,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        pin_memory_device=device,
        num_workers=1,
    )

    trainer = Trainer(
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
        output_interval=1000,
    )
    trainer.logger.info("Args: " + args.__str__())
    trainer.logger.info(f"Lmbda={lmbda:.12f}")

    if args.checkpoint or args.continue_train:  # load from previous checkpoint
        ckpt = (
            os.path.join(save_path, "last_epoch.pth.tar")
            if args.continue_train
            else args.checkpoint
        )
        trainer.logger.info(f"Loading {ckpt}")
        checkpoint = torch.load(ckpt, map_location=device)
        net.load_state_dict(checkpoint["state_dict"])
        if args.continue_train:
            trainer.load_state_dict(checkpoint["trainer"])

    while trainer.epoch < args.epochs:
        net.train()
        trainer.train_one_epoch(net)
        net.eval()
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
