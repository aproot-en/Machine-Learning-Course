

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_steps(steps, output_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(list(steps), list(steps.values()), color="steelblue")
    ax.set_ylabel("Rows remaining")
    ax.set_title("Rows remaining after each step")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    for i, value in enumerate(steps.values()):
        ax.text(i, value, str(value), ha="center", va="bottom", fontsize=9)

    save(fig, output_dir, "cleaning_steps.png")


def plot_before_after(before, after, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, data, title in zip(axes, [before, after], ["Before", "After"]):
        ax.hist(data, bins=40, color="steelblue")
        ax.set_xlabel("age")
        ax.set_title(f"{title} (n = {len(data)})")

    save(fig, output_dir, "before_after.png")


def save(fig, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

    print(f"Saved: {filename}")

