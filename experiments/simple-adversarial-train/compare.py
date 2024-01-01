from main import parse_args
import sys
import os
import torch
import argparse
import numpy as np
import random
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from compressai.utils.eval_model.__main__ import inference

from nets import load_model, load_lmbda
from dataset import Kodak
from grad_attack import attack
from lib.trainer import AverageMeter
from criterion import RateDistortionLoss

from main import generate_exp_name

ROOTDIR = os.path.split(__file__)[0]


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Simple adversarial training for LIC.")

    # model parameters
    parser.add_argument("model", type=str)
    parser.add_argument("-p", "--parameter_set", type=int, nargs='+')  # Always load pretrained models

    # Adversarial parameters
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--adv_steps", type=int, default=100)

    parser.add_argument("--seed", type=int, default=19260817)

    # dataset parameters
    args = parser.parse_args(argv)
    return args

@torch.no_grad()
def test(net: nn.Module, dataloader, adversarial=False, lmbda=1., epsilon=0.01, steps=100):
    net.eval()
    device = next(net.parameters()).device
    loss_dict = {}

    for i, d in enumerate(dataloader):
        print(f"Testing image #{i}", flush=True)
        if isinstance(d, torch.Tensor):
            d = d.to(device)
        else:
            d_new = []
            for each in d:
                d_new.append(each.to(device))
            d = tuple(d_new)
        
        if adversarial:
            criterion = RateDistortionLoss(lmbda)
            d = attack(net, d, criterion, steps, epsilon=epsilon).to(device)

        out_criterion = inference(net, d[0])

        for key, value in out_criterion.items():
            loss_dict.setdefault(key, AverageMeter())
            loss_dict[key].update(value)

        sample_log = ""
        for k, v in out_criterion.items():
            sample_log += f"{k}={v:.5f}\n"
        print(sample_log, flush=True)

    result = {k:v.avg for k,v in loss_dict.items()}

    # test_log = f"Testing results(average)"
    # for k, v in result.items():
    #     test_log += f" | {k}={v:.5f}"
    # print(test_log)
    return result

def plot(bpps, psnrs, labels, title, save_path):
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    for bpp, psnr, label in zip(bpps, psnrs, labels):
        ax.plot(bpp, psnr, label=label, marker='*')
    plt.title(title)
    ax.legend()
    ax.grid()
    ax.set_xlabel("BPP")
    ax.set_ylabel("PSNR")
    plt.savefig(save_path+'.pdf')
    plt.savefig(save_path+'.png', dpi=300)

def generat_plot_name(model, parameter_set, epsilon, adv_steps):
    return f"Compare_{model}_{''.join([str(x) for x in parameter_set])}_{epsilon}_{adv_steps}"


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    device = "cuda"

    bpp_anchor = []
    bpp_retrained = []
    bpp_adv_anchor = []
    bpp_adv_retrained = []

    psnr_anchor = []
    psnr_retrained = []
    psnr_adv_anchor = []
    psnr_adv_retrained = []

    for paramset in args.parameter_set:
        expname = generate_exp_name(args.model, paramset, 100, 100, 0.01, 100)
        ckptname = os.path.join(ROOTDIR, expname, "best_epoch.pth.tar")

        net_anchor = load_model(args.model, paramset)
        net = load_model(args.model, paramset)
        checkpoint = torch.load(ckptname, map_location=device)
        net.load_state_dict(checkpoint["state_dict"])

        net_anchor.cuda()
        net.cuda()

        lmbda = load_lmbda(args.model, paramset)

        test_dataset = Kodak(None)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, pin_memory_device=device)

        result_anchor = test(net_anchor, test_dataloader)
        result_retrained = test(net, test_dataloader)
        result_adv_anchor = test(net_anchor, test_dataloader, True, lmbda, args.epsilon, args.adv_steps)
        result_adv_retrained = test(net, test_dataloader, True, lmbda, args.epsilon, args.adv_steps)

        bpp_anchor.append(result_anchor['bpp'])
        bpp_retrained.append(result_retrained['bpp'])
        bpp_adv_anchor.append(result_adv_anchor['bpp'])
        bpp_adv_retrained.append(result_adv_retrained['bpp'])

        psnr_anchor.append(result_anchor['psnr'])
        psnr_retrained.append(result_retrained['psnr'])
        psnr_adv_anchor.append(result_adv_anchor['psnr'])
        psnr_adv_retrained.append(result_adv_retrained['psnr'])

    pltname = generat_plot_name(args.model, args.parameter_set, args.epsilon, args.adv_steps)
    plot([bpp_anchor, bpp_retrained], [psnr_anchor, psnr_retrained], [args.model, args.model+'*'], 'Kodak', os.path.join(ROOTDIR, pltname))
    plot([bpp_adv_anchor, bpp_adv_retrained], [psnr_adv_anchor, psnr_adv_retrained], [args.model, args.model+'*'], 'Kodak-adversary', os.path.join(ROOTDIR, pltname+'_adversary'))