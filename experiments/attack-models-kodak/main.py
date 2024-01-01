import argparse
import matplotlib.pyplot as plt
import torch

from dataset import Kodak
from nets import *
from grad_attack import attack
from criterion import RateDistortionLoss, PSNR
from utils import get_recon_bpp, jpeg_bpp_psnr_curve


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, choices=MODELS.keys())
    parser.add_argument("--quality", type=int, nargs='+', choices=[1,2,3,4,5,6], default=[1,2,3,4,5,6])
    parser.add_argument("--img_id", type=int, default=0)
    parser.add_argument("--lmbdas", type=float, nargs='+', default=[0., 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.004, 1.0])
    parser.add_argument("--epsilon", type=float, default=1e-2)
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("-o", type=str, required=True)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    ds = Kodak(None)
    img = ds[args.img_id]

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)

    bpp_anchors = []
    psnr_anchors = []

    jpeg_reference = jpeg_bpp_psnr_curve(img, 2.0)
    ax.plot(jpeg_reference[0], jpeg_reference[1], color='grey', label='JPEG')

    for quality in args.quality:
        net = load_model(args.net, quality=quality)
        print(f"Quality={quality}")

        _, bpp_an, psnr_an = get_recon_bpp(net, img)
        bpp_anchors.append(bpp_an)
        psnr_anchors.append(psnr_an)

        bpps = []
        psnrs = []

        min_psnr_origin = float('inf')

        for lmbda in args.lmbdas:
            criterion = RateDistortionLoss(lmbda)
            attk = attack(net=net, img=img, num_steps=args.num_steps, epsilon=args.epsilon, criterion=criterion)
            psnr_origin = PSNR(attk, img)
            print(f"PSNR-origin-attack: {psnr_origin:.5f}")
            if min_psnr_origin > psnr_origin:
                min_psnr_origin = psnr_origin
            _, bpp, psnr = get_recon_bpp(net, attk)
            bpps.append(bpp)
            psnrs.append(psnr)
            ax.plot([bpp_an, bpp], [psnr_an, psnr], color='b', linestyle='--', linewidth=0.4)

        ax.scatter(bpps, psnrs, color='b', marker='.', label='adversarial')

    ax.plot(bpp_anchors, psnr_anchors, color='r', label='origin', marker='*')
    plt.title(f"{args.net}; $\\epsilon = {args.epsilon}$; Quality={args.quality}")
    ax.set_xlabel("BPP")
    ax.set_ylabel("PSNR")
    ax.legend()
    plt.savefig(args.o+'.png', dpi=300)
    plt.savefig(args.o+'.pdf')