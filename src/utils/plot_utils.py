import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


save_path = os.path.abspath("../images")


def compute_metadata_accuracies(Y_hat, Y, metadata):
    correct = (np.argmax(Y_hat, axis=1) == np.argmax(Y, axis=1)) * 1
    accuracies = {i: 0 for i in metadata}
    counts = {i: 0 for i in metadata}
    for i in range(Y_hat.shape[0]):
        accuracies[metadata[i]] += correct[i]
        counts[metadata[i]] += 1
    for k in accuracies.keys():
        accuracies[k] /= counts[k]
    return accuracies


def compute_metadata_output_avgs(Y, metadata):
    metadata_unique = np.unique(metadata)
    output_avgs = np.zeros((len(metadata_unique), Y.shape[1]))
    output_stds = np.zeros((len(metadata_unique), Y.shape[1]))
    for (i, metadata_bucket) in enumerate(metadata_unique):
        output_avgs[i] = np.mean(Y[metadata == metadata_bucket], axis=0)
        output_stds[i] = np.std(Y[metadata == metadata_bucket], axis=0)

    return metadata_unique, output_avgs, output_stds


def plot_fit_history(fit_history, save_as=None):
    matplotlib.rcParams.update({"font.size": 12})

    n_epochs = len(fit_history["accuracy"])
    x = range(1, n_epochs + 1)

    fig, ax = plt.subplots(2, 1, figsize=(9, 9), dpi=144)
    ax[0].plot(x, fit_history["accuracy"], c="blue", linewidth=1, alpha=0.8, label="Train")
    ax[0].plot(
        x, fit_history["val_accuracy"], c="red", linewidth=1, alpha=0.8, label="Validation"
    )
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Accuracy")
    ax[0].set_yticks(np.linspace(0, 1, 11))
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(x, fit_history["loss"], c="blue", linewidth=1, alpha=0.8, label="Train")
    ax[1].plot(
        x, fit_history["val_loss"], c="red", linewidth=1, alpha=0.8, label="Validation"
    )
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")
    ax[1].set_yticks(np.linspace(0, 0.5, 6))
    ax[1].grid()

    plt.tight_layout()

    if save_as:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, save_as))

    return fig


def plot_lattice(lattice, title=None, save_as=False):
    matplotlib.rcParams.update({"font.size": 12})

    fig, ax = plt.subplots(dpi=144, figsize=(6, 6))
    ax.matshow(lattice, cmap="gray_r", aspect="equal")
    ax.set(xticks=[], yticks=[])
    if title:
        ax.set_title(title)
    plt.tight_layout()

    if save_as:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, save_as))


def plot_p_accuracies(Y_hat, Y, p, save_as=False):
    matplotlib.rcParams.update({"font.size": 12})

    accuracies = compute_metadata_accuracies(Y_hat, Y, p)

    fig, ax = plt.subplots(1, dpi=144, figsize=(9, 6))
    ax.plot(*zip(*sorted(accuracies.items())), c="b", alpha=0.8)
    ax.set_xlabel(r"$p$")
    ax.set_ylabel(r"Accuracy")
    ax.set_ylim((0.75, 1.05))
    ax.set_xticks(np.linspace(0, 1, 11))
    ax.set_yticks(np.linspace(0.8, 1, 5))
    ax.grid()
    plt.tight_layout()

    if save_as:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, save_as))


def plot_p_output_avgs(Y, p, save_as=None):
    matplotlib.rcParams.update({"font.size": 12})

    p_unique, output_avgs, output_stds = compute_metadata_output_avgs(Y, p)

    fig, ax = plt.subplots(1, dpi=144, figsize=(9, 6))
    ax.plot(
        p_unique,
        output_avgs[:, 0],
        c="b",
        linewidth=1,
        alpha=0.8,
        label=r"$y_0$",
        marker="o",
    )
    ax.plot(
        p_unique,
        output_avgs[:, 1],
        c="r",
        linewidth=1,
        alpha=0.8,
        label=r"$y_1$",
        marker="o",
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
