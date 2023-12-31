import argparse
import matplotlib.pyplot as plt

from dataset import Kodak
from nets import *
from grad_attack import attack


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, choices=MODELS.keys())
    parser.add_argument("quality", type=int, choices=[1,2,3,4,5,6])
    parser.add_argument("--img_id", type=int, default=0)
    parser.add_argument("--lmbdas", type=float, nargs='+', default=[0., 0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--num_steps", type=int, default=100)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    net = load_model(args.net, quality=args.quality)
    ds = Kodak(None)
    img = ds[args.img_id]

    for lmbda in args.lmbdas:
        attk = attack(net=net, img=img, num_steps=args.num_steps, epsilon=args.epsilon, lmbda=args.lmbda)