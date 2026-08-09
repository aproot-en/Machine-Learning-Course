import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def plot_new_feature(counts, output_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index.astype(str), counts.values, color="darkorange")
    ax.set_ylabel("count")
    ax.set_title("age_group - the new feature")

    save(fig, output_dir, "new_feature.png")


def plot_scores(names, scores, output_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(names, scores, color="seagreen")
    ax.set_xlabel("score")
    ax.set_title("Feature importance (SelectKBest)")

    save(fig, output_dir, "feature_scores.png")


def save(fig, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

    print(f"Saved: {filename}")
