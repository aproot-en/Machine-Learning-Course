

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_transforms(results, output_dir):
    fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 3.6))

    for ax, (name, values) in zip(axes, results.items()):
        ax.hist(values, bins=40, color="steelblue")
        ax.set_title(name)
    save(fig, output_dir, "transforms.png")


def plot_skew(names, skews, output_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, skews, color=["tomato" if abs(s) > 0.5 else "seagreen"
                                for s in skews])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("skewness")
    ax.set_title("Skewness (near 0 = symmetric)")

    save(fig, output_dir, "skewness.png")


def save(fig, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

    print(f"Saved: {filename}")

