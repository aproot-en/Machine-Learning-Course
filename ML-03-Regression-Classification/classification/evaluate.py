import matplotlib

# Set backend before pyplot, so it works without a display
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


def evaluate_model(y_test, predictions, classes, save_path):

    # Pin label order so target_names always matches the columns
    labels = list(range(len(classes)))

    accuracy = accuracy_score(y_test, predictions)

    print("\n------------ Evaluation ------------------")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, labels=labels,
                                target_names=classes, zero_division=0))

    matrix = confusion_matrix(y_test, predictions, labels=labels)
    print("Confusion Matrix:")
    print(matrix)

    plot_confusion_matrix(matrix, classes, save_path)
    print(f"Saved: {save_path}")

    return accuracy


def plot_confusion_matrix(matrix, classes, save_path):

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(matrix, cmap="Blues")

    ax.set_xticks(np.arange(len(classes)), classes)
    ax.set_yticks(np.arange(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    threshold = matrix.max() / 2
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, matrix[i, j], ha="center", va="center",
                    color="white" if matrix[i, j] > threshold else "black")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_samples(images, y_test, predictions, classes, save_path, n_samples=8):

    index = np.random.choice(len(images), n_samples, replace=False)

    fig, axes = plt.subplots(2, 4, figsize=(10, 6))
    axes = axes.ravel()

    for ax, i in zip(axes, index):
        correct = predictions[i] == y_test[i]

        ax.imshow(images[i], cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Pred {classes[predictions[i]]}\n"
                     f"True {classes[y_test[i]]}",
                     color="green" if correct else "red")

    fig.suptitle("Gender prediction")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")
