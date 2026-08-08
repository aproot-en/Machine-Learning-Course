import matplotlib

# Set backend before pyplot, so it works without a display
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(y_test, predictions, save_path):

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("\n------------ Evaluation ------------------")
    print(f"MAE  : {mae:.2f} years")
    print(f"RMSE : {rmse:.2f} years")
    print(f"R2   : {r2:.4f}")

    plot_results(y_test, predictions, save_path)
    print(f"Saved: {save_path}")

    return mae


def plot_results(y_test, predictions, save_path):

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].scatter(y_test, predictions, s=6, alpha=0.25)
    # A perfect model would put every point on this diagonal
    axes[0].plot([0, 116], [0, 116], "r--")
    axes[0].set_xlabel("Actual age")
    axes[0].set_ylabel("Predicted age")
    axes[0].set_title("Predicted vs Actual")

    axes[1].hist(predictions - y_test, bins=60)
    axes[1].axvline(0, color="r", linestyle="--")
    axes[1].set_xlabel("Predicted - Actual (years)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residuals")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_samples(images, y_test, predictions, save_path, n_samples=8):

    index = np.random.choice(len(images), n_samples, replace=False)

    fig, axes = plt.subplots(2, 4, figsize=(10, 6))
    axes = axes.ravel()

    for ax, i in zip(axes, index):
        correct = abs(predictions[i] - y_test[i]) <= 5

        ax.imshow(images[i], cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Pred {predictions[i]:.0f}\nTrue {y_test[i]}",
                     color="green" if correct else "red")

    fig.suptitle("Age prediction (green = within 5 years)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")
