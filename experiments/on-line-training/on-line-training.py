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
from grad_attack import adversarial_tests, attack
from criterion import RateDistortionLoss

ROOTDIR = os.path.split(__file__)[0]

def online_training_wrapper(net, step, criterion):
    compress_old = net.compress
    def compress_new(x):
        x = attack(net=net, img=x, criterion=criterion, num_steps=step, epsilon=1., inverse=True)
        return compress_old(x)
    net.compress = compress_new
    return net

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Simple adversarial training for LIC.")

    # model parameters
    parser.add_argument("model", type=str)
    parser.add_argument("-p", "--parameter_set", type=int, nargs='+')  # Always load pretrained models
    parser.add_argument("--train_steps", type=int, default=1000)

    # Adversarial parameters
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--adv_steps", type=int, default=100)

    parser.add_argument("--seed", type=int, default=19260817)

    # dataset parameters
    args = parser.parse_args(argv)
    return args

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

def generat_plot_name(train_steps, model, parameter_set, epsilon, adv_steps):
    return f"Online_training_{train_steps}_{model}_{''.join([str(x) for x in parameter_set])}_{epsilon}_{adv_steps}"

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
    bpp_adv_ideal_anchor = []
    bpp_adv_ideal_retrained = []

    psnr_anchor = []
    psnr_retrained = []
    psnr_adv_anchor = []
    psnr_adv_retrained = []
    psnr_adv_ideal_anchor = []
    psnr_adv_ideal_retrained = []

    for paramset in args.parameter_set:
        lmbda = load_lmbda(args.model, paramset)
        criterion = RateDistortionLoss(lmbda)

        net_anchor = load_model(args.model, paramset)
        net = load_model(args.model, paramset)
        net = online_training_wrapper(net, args.train_steps, criterion)

        net_anchor.cuda()
        net.cuda()

        test_dataset = Kodak(None)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, pin_memory_device=device)

        result_anchor, result_adv_anchor, result_adv_ideal_anchor = adversarial_tests(net_anchor, test_dataloader, lmbda, args.epsilon, args.adv_steps)

        result_retrained, result_adv_retrained, result_adv_ideal_retrained = adversarial_tests(net, test_dataloader, lmbda, args.epsilon, args.adv_steps)

        bpp_anchor.append(result_anchor['bpp'])
        bpp_retrained.append(result_retrained['bpp'])
        bpp_adv_anchor.append(result_adv_anchor['bpp'])
        bpp_adv_retrained.append(result_adv_retrained['bpp'])
        bpp_adv_ideal_anchor.append(result_adv_ideal_anchor['bpp'])
        bpp_adv_ideal_retrained.append(result_adv_ideal_retrained['bpp'])

        psnr_anchor.append(result_anchor['psnr'])
        psnr_retrained.append(result_retrained['psnr'])
        psnr_adv_anchor.append(result_adv_anchor['psnr'])
        psnr_adv_retrained.append(result_adv_retrained['psnr'])
        psnr_adv_ideal_anchor.append(result_adv_ideal_anchor['psnr'])
        psnr_adv_ideal_retrained.append(result_adv_ideal_retrained['psnr'])

    pltname = generat_plot_name(args.train_steps, args.model, args.parameter_set, args.epsilon, args.adv_steps)
    plot([bpp_anchor, bpp_retrained, bpp_adv_anchor, bpp_adv_retrained, bpp_adv_ideal_anchor, bpp_adv_ideal_retrained], [psnr_anchor, psnr_retrained, psnr_adv_anchor, psnr_adv_retrained, psnr_adv_ideal_anchor, psnr_adv_ideal_retrained], [args.model, args.model+'*', args.model+"(adv)", args.model+'*(adv)', args.model+"(adv-ideal)", args.model+'*(adv-ideal)'], 'Kodak', os.path.join(ROOTDIR, pltname))