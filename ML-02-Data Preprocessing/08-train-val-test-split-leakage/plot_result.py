

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def plot_split_sizes(sizes, output_dir):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.pie(sizes.values(), labels=list(sizes),
           autopct=lambda p: f"{p:.0f}%\n({int(p * sum(sizes.values()) / 100)})",
           colors=["seagreen", "gold", "tomato"])
    ax.set_title("Split proportions")

    save(fig, output_dir, "split_sizes.png")


def plot_balance(distributions, classes, output_dir):
    """Compare the class distribution in each set - they should be similar"""
    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(distributions)
    width = 0.35
    x = np.arange(len(names))

    for i, label in enumerate(classes):
        values = [distributions[n][i] * 100 for n in names]
        ax.bar(x + i * width, values, width, label=label)

    ax.set_xticks(x + width / 2, names)
    ax.set_ylabel("%")
    ax.set_title("Class balance per split (stratify keeps them equal)")
    ax.legend()

    save(fig, output_dir, "class_balance.png")


def save(fig, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

    print(f"Saved: {filename}")
