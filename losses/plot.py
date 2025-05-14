import matplotlib.pyplot as plt

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


def plot_imgs(*args):
    args = list(map(lambda x: x.cpu().detach().numpy(), args))

    fig, axs = plt.subplots(nrows=1, ncols=len(args))
    for ax, weight in zip(axs, args):
        ax.imshow(weight, cmap='grey')

    for ax in axs:
        ax.set_axis_off()

    plt.show(block=True)