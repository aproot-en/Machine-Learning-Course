

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_boxplot(series, output_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(series, vert=False)
    ax.set_xlabel("age")
    ax.set_title("Boxplot - points outside the whiskers are outliers")

    save(fig, output_dir, "boxplot.png")


def plot_outliers(series, low, high, output_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(series, bins=50, color="steelblue")
    ax.axvline(low, color="red", linestyle="--", label=f"lower = {low:.1f}")
    ax.axvline(high, color="red", linestyle="--", label=f"upper = {high:.1f}")
    ax.set_xlabel("age")
    ax.set_ylabel("count")
    ax.set_title("IQR bounds")
    ax.legend()

    save(fig, output_dir, "outlier_bounds.png")

def save(fig, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

    print(f"Saved: {filename}")
