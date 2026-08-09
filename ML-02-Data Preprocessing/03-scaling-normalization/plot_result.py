import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def plot_scalers(results, output_dir):
    """เทียบการกระจายตัวก่อนและหลัง scale"""

    fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 3.6))

    for ax, (name, values) in zip(axes, results.items()):
        ax.hist(values, bins=40, color="steelblue")
        ax.set_title(f"{name}\nmin={values.min():.2f}  max={values.max():.2f}")

    save(fig, output_dir, "scalers.png")


def plot_range(results, output_dir):
    """แท่งเทียบช่วงค่า min-max ของแต่ละวิธี"""

    fig, ax = plt.subplots(figsize=(7, 4))

    names = list(results)
    ax.bar(names, [v.max() - v.min() for v in results.values()], color="teal")
    ax.set_ylabel("max - min")
    ax.set_title("Value range after scaling")
    ax.set_yscale("log")

    save(fig, output_dir, "value_range.png")


def save(fig, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

    print(f"Saved: {filename}")
