

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_missing(df, output_dir):
    counts = df.isna().sum()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index, counts.values, color="tomato")
    ax.set_ylabel("Missing values")
    ax.set_title("Missing values per column")

    save(fig, output_dir, "missing_per_column.png")


def plot_compare(original, dropped, filled, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, data, title in zip(
        axes,
        [original, dropped, filled],
        ["Original (with NaN)", "dropna()", "fillna(median)"],
    ):
        ax.hist(data.dropna(), bins=40, color="steelblue")
        ax.set_title(f"{title}\nn = {data.notna().sum()}")
        ax.set_xlabel("age")

    save(fig, output_dir, "compare_methods.png")


def save(fig, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

    print(f"Saved: {filename}")
