

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_file_sizes(sizes, output_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(list(sizes), [v / 1024 for v in sizes.values()], color="cornflowerblue")
    ax.set_ylabel("KB")
    ax.set_title("File size - same 200 rows in each format")

    save(fig, output_dir, "file_sizes.png")


def plot_image(image, output_dir):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image, cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Read back from .png")
    save(fig, output_dir, "image_read_back.png")

def save(fig, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

    print(f"Saved: {filename}")
