
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from load_data import load_data, OUTPUT_DIR
from plot_result import plot_split_sizes, plot_balance

CLASSES = ["Male", "Female"]


def main():
    print("-" * 30)
    print("08 - Train-Validation-Test Split and Data Leakage")
    print("-" * 30)

    df = load_data()
    X = df[["age", "ethnicity"]]
    y = df["gender"]

    print("\n[1] split into 3 sets: 70-10-20")

    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=0.125, random_state=42, stratify=y_rest
    )

    sizes = {"train": len(X_train), "validation": len(X_val), "test": len(X_test)}
    for name, size in sizes.items():
        print(f"  {name:<12}{size:>7} ({size / len(X) * 100:.0f}%)")

    print("\n[2] what each set is used for")
    print("train      - to train the model")
    print("validation - to tune hyperparameters and select the best model")
    print("test       - to evaluate the final model performance")

    print("\n[3] stratify ensures balanced class distribution across all sets")
    distributions = {}
    for name, part in [("train", y_train), ("validation", y_val), ("test", y_test)]:
        ratio = np.bincount(part, minlength=2) / len(part)
        distributions[name] = ratio
        print(f"  {name:<12}Male {ratio[0] * 100:.1f}%  Female {ratio[1] * 100:.1f}%")

    print("\n[4] Data Leakage - data test is leaking into the training process")

    print("\nWrong approach: fit scaler before splitting")
    wrong = StandardScaler().fit(X)
    print(f" mean is scaler = {wrong.mean_.round(3)}")
    print("  valueis calculated from all data, including the test set")
    print("  the result is that the test score will be overly optimistic")

    print("\nCorrect: fit only on train")
    right = StandardScaler().fit(X_train)
    print(f" mean is scaler = {right.mean_.round(3)}")
    print("  the result is that the test score will be overly optimistic")

    print("\n[5] Leakage is even more subtle and can occur in various ways")
    print("- filling missing values with the overall mean")
    print("- removing duplicate rows after splitting, causing the same images to appear in both train and test")
    print("- using future data to predict past events in time-series tasks")

    print("\n[6] Summary")
    plot_split_sizes(sizes, OUTPUT_DIR)
    plot_balance(distributions, CLASSES, OUTPUT_DIR)


if __name__ == "__main__":
    main()
