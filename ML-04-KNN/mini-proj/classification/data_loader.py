


"""
Read CSV
convert text to number
make Scaling for KNN
split data: train / validation / test
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CSV_PATH = Path(__file__).resolve().parent.parent / "data-animal" / "animal_dataset.csv"

TARGET = "Diet_Type"

# config feature is numeric features that are already numbers
NUMERIC_FEATURES = [
    "Average_Lifespan (Years)",
    "Weight (kg)",
    "Height (cm)",
    "Speed (km/h)",
    "Endangered_Level (1-5)",
]

# config feature is text features that need to be converted to numbers
TEXT_FEATURES = {
    "Intelligence_Level": {"Low": 0, "Medium": 1, "High": 2},
    "Domesticated": {"No": 0, "Yes": 1},
    "Nocturnal": {"No": 0, "Yes": 1},
}






# ---------------------------------------------------------------------------
def load_data(test_size=0.2, seed=42):

    # step 1 : read CSV
    df = pd.read_csv(CSV_PATH)
    df = df.dropna()            

    # step 2 : convert text to number
    X = df[NUMERIC_FEATURES].copy()
    for col, mapping in TEXT_FEATURES.items():
        X[col] = df[col].map(mapping)      # เช่น "High" -> 2

    # convert result (target) to number : Carnivore->0, Herbivore->1, Omnivore->2
    class_names = sorted(df[TARGET].unique())
    y = df[TARGET].map({name: i for i, name in enumerate(class_names)})

    X = X.to_numpy(dtype="float32")
    y = y.to_numpy(dtype="int32")

    # step 3 : split data เป็น train 60 / validation 20 / test 20
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=seed, stratify=y_temp)

    # step 4 : Scaling 
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype("float32")
    X_val = scaler.transform(X_val).astype("float32")
    X_test = scaler.transform(X_test).astype("float32")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "class_names": class_names,
        "feature_names": NUMERIC_FEATURES + list(TEXT_FEATURES),
        "n_rows": len(df),
    }


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data = load_data()
    print("train :", data["X_train"].shape)
    print("val   :", data["X_val"].shape)
    print("test  :", data["X_test"].shape)
    print("คลาส  :", data["class_names"])
