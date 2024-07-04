import matplotlib.pyplot as plt

from lib.path import *

def plot(bpps, psnrs, labels, title, save_path):
    make_folder(save_path)
    fig = plt.figure(figsize=(9, 6))
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
    plt.close()
