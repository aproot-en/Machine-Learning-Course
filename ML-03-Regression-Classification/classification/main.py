import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split

# data_loader.py is one level up, shared with regression/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, ROOT_DIR)

from data_loader import load_data, to_features, as_images
from model import train_model, predict_model
from evaluate import evaluate_model, plot_samples

CSV_PATH = os.path.join(ROOT_DIR, "age_gender.csv")
OTHERS_DIR = os.path.join(ROOT_DIR, "others_dir")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

CLASSES = ["Male", "Female"]   # 0 = Male, 1 = Female
TEST_SIZE = 0.2
PCA_COMPONENTS = 100
C = 1.0


def main():

    print("--" * 30)
    print("Classification: GENDER prediction")
    print("--" * 30)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Read the spreadsheet
    print("\n[Step 1] Reading age_gender.csv...")
    pixels, meta = load_data(CSV_PATH, OTHERS_DIR)

    print(f"Rows    : {len(pixels)}")
    for label, count in meta["gender"].value_counts().sort_index().items():
        print(f"  {CLASSES[label]:<7}: {count}")

    # Step 2: Preprocessing
    print("\n[Step 2] Building feature matrix...")
    X = to_features(pixels)
    y = meta["gender"].to_numpy()
    print(f"Feature shape: {X.shape}")

    # Step 3: Split Dataset
    print("\n[Step 3] Splitting dataset...")
    index = np.arange(len(X))
    X_train, X_test, y_train, y_test, _, test_index = train_test_split(
        X, y, index, test_size=TEST_SIZE, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

    # Step 4: Train
    print("\n[Step 4] Training logistic regression...")
    model = train_model(X_train, y_train, PCA_COMPONENTS, C)
    print("Training completed.")

    # Step 5: Prediction
    print("\n[Step 5] Testing model...")
    predictions = predict_model(model, X_test)

    # Step 6: Evaluation
    print("\n[Step 6] Evaluating model...")
    evaluate_model(y_test, predictions, CLASSES,
                   os.path.join(OUTPUT_DIR, "confusion_matrix.png"))

    plot_samples(as_images(pixels[test_index]), y_test, predictions, CLASSES,
                 os.path.join(OUTPUT_DIR, "gender_samples.png"))


if __name__ == "__main__":
    main()


