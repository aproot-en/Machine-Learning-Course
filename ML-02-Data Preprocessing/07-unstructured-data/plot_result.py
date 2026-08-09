

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_faces(images, meta, output_dir):

    fig, axes = plt.subplots(2, 4, figsize=(10, 6))
    for ax, i in zip(axes.ravel(), range(8)):
        ax.imshow(images[i], cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"age {meta['age'][i]}")

    fig.suptitle("Unstructured data - 48x48 images from the pixels column")
    save(fig, output_dir, "faces.png")


def plot_pixel_hist(images, output_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(images.ravel(), bins=50, color="gray")
    ax.set_xlabel("pixel value (0-255)")
    ax.set_ylabel("count")
    ax.set_title("Pixel value distribution")

    save(fig, output_dir, "pixel_histogram.png")


def save(fig, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

    print(f"Saved: {filename}")

