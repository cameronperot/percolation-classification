import os
import numpy as np
from tensorflow import keras
import matplotlib
import matplotlib.pyplot as plt


data_path = os.path.abspath("../data")
model_path = os.path.abspath("../artifacts")
save_path = os.path.abspath("../images")

data = {}
model = {}
Y_hat = {}

Ls = [4, 8, 16, 32]
for L in Ls:
    data[L] = Data(os.path.join(data_path, "data_transition_%i.npz" % L), test_ratio=1)
    model[L] = keras.models.load_model(os.path.join(model_path, ("model_%i" % L) + ".h5"))
    Y_hat[L] = model[L].predict(data[L].test.X)


# %%


def compute_metadata_output_avgs(Y_hat, metadata):
    metadata_unique = np.unique(metadata)
    output_avgs = np.zeros((len(metadata_unique), Y_hat.shape[1]))
    output_stds = np.zeros((len(metadata_unique), Y_hat.shape[1]))
    for (i, metadata_bucket) in enumerate(metadata_unique):
        output_avgs[i] = np.mean(Y_hat[metadata == metadata_bucket], axis=0)
        output_stds[i] = np.std(Y_hat[metadata == metadata_bucket], axis=0)

    return metadata_unique, output_avgs, output_stds


def plot_p_output_avgs(data, Y_hat, save_as=None):
    matplotlib.rcParams.update({"font.size": 12})

    colors = ["r", "g", "b", "darkorange", "purple"]
    markers = ["o", "^", "s", "D", "d"]
    fig, ax = plt.subplots(1, dpi=144, figsize=(9, 6))

    for (i, L) in enumerate(Ls):
        p_unique, output_avgs, output_stds = compute_metadata_output_avgs(
            Y_hat[L], data[L].test.p
        )
        ax.plot(
            p_unique,
            output_avgs[:, 0],
            c=colors[i],
            linewidth=1,
            alpha=0.8,
            label=r"$L = %i$" % L,
            marker=markers[i],
        )
        ax.plot(
            p_unique,
            output_avgs[:, 1],
            c=colors[i],
            linewidth=1,
            alpha=0.8,
            marker=markers[i],
        )
    ax.set_xlabel(r"$p$")
    ax.set_ylabel(r"$y_0, y_1$")
    ax.set_ylim((-0.1, 1.1))
    ax.set_xticks(np.linspace(0.5, 0.65, 16))
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.axvline(x=0.59274, c="k", label=r"$p_c$")
    ax.grid()
    ax.legend()

    plt.tight_layout()

    if save_as:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, save_as))

    return fig


plot_p_output_avgs(data, Y_hat, save_as="transition.png")
