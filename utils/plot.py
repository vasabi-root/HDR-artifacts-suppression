import matplotlib.pyplot as plt
from typing import List
import numpy as np
import torch

def plot_weights(LY, l_map, cs_map):
    LY, l_map, cs_map = map(lambda x: x.cpu().detach().numpy(), [LY, l_map, cs_map])

    fig, axs = plt.subplots(nrows=3, ncols=len(LY))
    for ax, weight in zip(axs[0], LY):
        ax.imshow(weight, cmap='grey')
    axs[1][len(LY) // 2].imshow(l_map, cmap='grey')
    axs[2][len(LY) // 2].imshow(cs_map, cmap='grey')

    for ax_row in axs:
        for ax in ax_row:
            ax.set_axis_off()

    plt.show(block=True)


def plot_imgs(*args, save_name=None):
    args = list(map(lambda x: x.cpu().detach().numpy(), args))

    fig, axs = plt.subplots(nrows=1, ncols=len(args))
    if len(args) == 1:
        axs = [axs]
    for ax, img in zip(axs, args):
        img = img.squeeze()
        if len(img) == 3:
            img = img.transpose(1, 2, 0)
            ax.imshow(img)
        else:
            ax.imshow(img, cmap='grey')
        ax.set_axis_off()     

    if save_name:
        fig.savefig(save_name, dpi=300)
    else:
        plt.show(block=True)

def plot_bunches(*args, save_name=None):
    # args = list(map(lambda x: x.cpu().detach().numpy(), args))
    n = len(args[0])
    for arg in args:
        assert len(arg) == n

    fig, axs = plt.subplots(nrows=len(args), ncols=n)
    if len(args) == 1:
        axs = [axs]
    for ax_row, bunch in zip(axs, args):
        for ax, img in zip(ax_row, bunch):
            img = img.squeeze()
            if isinstance(img, torch.Tensor):
                img = img.cpu().detach().numpy()
            if len(img) == 3:
                img = img.transpose(1, 2, 0)
                ax.imshow(img)
            else:
                ax.imshow(img, cmap='grey')
            ax.set_axis_off()

    if save_name:
        fig.savefig(save_name, dpi=500)
    else:
        plt.show(block=True)

def plot_luts(luts: torch.Tensor, save_name):
    luts_np = luts.cpu().detach().numpy()
    fig, axs = plt.subplots(ncols=len(luts_np), figsize=(12, 6))

    for ax, lut in zip(axs, luts_np):
        ax.plot(range(256), lut)
        ax.set_xlim(0, 256)
        ax.set_ylim(0.0, 1.0)
        ax.grid()
    plt.tight_layout()
    fig.savefig(save_name, dpi=300)