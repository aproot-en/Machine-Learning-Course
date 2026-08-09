


import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_counts(counts, labels, output_dir):
    #Plot bar chart of category counts
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, counts, color="mediumpurple")
    ax.set_ylabel("count")
    ax.set_title("Count per ethnicity")

    for i, value in enumerate(counts):
        ax.text(i, value, str(value), ha="center", va="bottom")

    save(fig, output_dir, "category_counts.png")


def plot_onehot(encoded, labels, output_dir):
    #Plot one-hot encoded matrix for the first 10 rows

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(encoded[:10], cmap="Blues", aspect="auto")

    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(10), [f"row {i}" for i in range(10)])
    ax.set_title("One-hot encoding (first 10 rows)")

    save(fig, output_dir, "onehot.png")


def save(fig, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

    print(f"Saved: {filename}")
