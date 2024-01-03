import sys
import os
import torch
import argparse
import numpy as np
import random
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt

from nets import load_model, load_lmbda
from dataset import Kodak
from grad_attack import adversarial_test

from main import generate_exp_name
from lib.path import *
from lib.bdrate import BD_RATE

ROOTDIR = os.path.split(__file__)[0]

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Simple adversarial training for LIC.")

    # model parameters
    parser.add_argument("model", type=str)
    parser.add_argument("-p", "--parameter_set", type=int, nargs='+')  # Always load pretrained models

    # Adversarial parameters
    parser.add_argument("--epsilon_train", type=float, default=0.01)
    parser.add_argument("--adv_steps_train", type=int, default=100)

    parser.add_argument("--seed", type=int, default=19260817)

    # dataset parameters
    args = parser.parse_args(argv)
    return args

def plot(bpps, psnrs, labels, title, save_path):
    make_folder(save_path)
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

def generat_plot_name(model, parameter_set):
    return f"BPP-PSNR_{model}_{''.join([str(x) for x in parameter_set])}"


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    pltname = generat_plot_name(args.model, args.parameter_set)
    jsonname = os.path.join(ROOTDIR, "data", pltname+'.json')
    if os.path.isfile(jsonname):
        data = load_json(jsonname)
    else:
        device = "cuda"

        bpp_anchor = []
        bpp_retrained = []

        psnr_anchor = []
        psnr_retrained = []

        for paramset in args.parameter_set:
            expname = generate_exp_name(args.model, paramset, 100, 100, args.epsilon_train, args.adv_steps_train)
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

            result_anchor, _ = adversarial_test(net_anchor, test_dataloader)

            result_retrained, _ = adversarial_test(net, test_dataloader)

            bpp_anchor.append(result_anchor['bpp'])
            bpp_retrained.append(result_retrained['bpp'])

            psnr_anchor.append(result_anchor['psnr'])
            psnr_retrained.append(result_retrained['psnr'])
        
        data = {
            "bpp_anchor": bpp_anchor,
            "bpp_retrained": bpp_retrained,
            "psnr_anchor": psnr_anchor,
            "psnr_retrained": psnr_retrained,
        }

        dump_json(data, jsonname)

    bdrate = BD_RATE(data['bpp_anchor'], data['psnr_anchor'], data['bpp_retrained'], data['psnr_retrained'])
    plot([data['bpp_anchor'], data['bpp_retrained']], [data['psnr_anchor'], data['psnr_retrained']], [args.model, args.model+'+Adv.'], 'Kodak', os.path.join(ROOTDIR, "plots", pltname))
    print(f"BD-rate={bdrate:.3f}%")