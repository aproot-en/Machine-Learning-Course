##















import matplotlib
matplotlib.use("Agg")    

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


# ---------------------------------------------------------------------------
def plot_k_curve(k_values, scores, out_path):

    plt.figure(figsize=(7, 4.5))
    plt.plot(k_values, scores, "o-")
    plt.xlabel("k (number of neighbors)")
    plt.ylabel("Validation accuracy")
   # plt.title("Choosing the best k")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, class_names, out_path):
    """confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(class_names)), class_names, rotation=30)
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True label")
 #   plt.title("Confusion Matrix")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    return cm


# ---------------------------------------------------------------------------
def print_report(y_true, y_pred, class_names):
    """show (precision / recall / f1)"""
    print(classification_report(y_true, y_pred,
                                target_names=class_names, zero_division=0))


# ---------------------------------------------------------------------------
def save_predictions(y_true, y_pred, class_names, out_path):
    """save CSV"""
    df = pd.DataFrame({
        "true_label": [class_names[i] for i in y_true],
        "predicted_label": [class_names[i] for i in y_pred],
        "correct": y_true == y_pred,
    })
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return df
