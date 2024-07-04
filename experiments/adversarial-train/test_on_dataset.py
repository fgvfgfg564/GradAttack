import torch
from torch.utils.data import DataLoader
import os
import sys
import argparse
import numpy as np

from lib.test import test_on_dataset
from lib.path import *
from lib.utils import load_weights
from dataset import *
from nets import load_model

ROOTDIR = os.path.split(__file__)[0]

from train_baseline import add_key_args, generate_exp_name
from train_adversarial import generate_adv_exp_name, parse_args


def generate_test_exp_name(args):
    return (
        f"test/{args.model}/{args.test_dataset}--{args.dataset}-{args.epochs}x{args.steps_per_epoch}"
        f"-{args.steps}--{args.adv_steps}-{args.adv_optimizer}-{args.adv_lr}-{args.adv_epsilon}/{args.lmbda}"
    )


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    lmbdas = [0.0018, 0.0067, 0.0250, 0.0932]

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    for lmbda in lmbdas:
        args.lmbda = lmbda
        save_path = os.path.join(ROOTDIR, generate_test_exp_name(args))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        base_path = os.path.join(ROOTDIR, generate_exp_name(args))
        adv_path = os.path.join(ROOTDIR, generate_adv_exp_name(args))
        base_ckpt = os.path.join(base_path, "best_epoch.pth.tar")
        adv_ckpt = os.path.join(adv_path, "best_epoch.pth.tar")

        device = "cuda"

        net_base = load_model(args.model).to(device)
        net_adv = load_model(args.model).to(device)
        lmbda = args.lmbda

        # load weights
        net_base.load_state_dict(load_weights(base_ckpt, device))
        net_adv.load_state_dict(load_weights(adv_ckpt, device))

        test_dataset = load_dataset(args.test_dataset)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            pin_memory_device=device,
            num_workers=1,
        )

        # Test adversarial performance

        net_base.eval()
        net_adv.eval()
        performance_baseline = test_on_dataset(net_base, test_dataloader)
        performance_new = test_on_dataset(net_adv, test_dataloader)

        dump_json(performance_baseline, os.path.join(save_path, "baseline.json"))
        dump_json(performance_new, os.path.join(save_path, "adversarial.json"))
