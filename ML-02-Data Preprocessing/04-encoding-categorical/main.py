



import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from load_data import load_data, OUTPUT_DIR
from plot_result import plot_counts, plot_onehot

ETHNICITY = ["White", "Black", "Asian", "Indian", "Other"]

def main():
    print("-" * 30)
    print("04 - Encoding Categorical Data")
    print("-" * 30)

    df = load_data()
    df["ethnicity_name"] = df["ethnicity"].map(dict(enumerate(ETHNICITY)))

    print("\n[1] Categorical Data")
    counts = df["ethnicity_name"].value_counts()
    print(counts)

    print("\n[2] Label Encoding - change category names to numbers")
    encoder = LabelEncoder()
    df["label"] = encoder.fit_transform(df["ethnicity_name"])

    for name, code in zip(encoder.classes_, range(len(encoder.classes_))):
        print(f"  {name:<8} -> {code}")

    print("\nDisadvantage: The model will think that 3 is greater than 1, even though they are just category names without an order")

    print("\n[3] One-Hot Encoding - each category gets its own column")
    onehot = OneHotEncoder(sparse_output=False)
    encoded = onehot.fit_transform(df[["ethnicity_name"]])
    labels = list(onehot.categories_[0])

    print(f"from 1 column becomes {encoded.shape[1]} columns: {labels}")
    print("\n5 rows:")
    print(pd.DataFrame(encoded[:5], columns=labels).astype(int).to_string(index=False))

    print("\n[4] pd.get_dummies - shortcut for pandas")
    dummies = pd.get_dummies(df["ethnicity_name"], prefix="eth")
    print(f"columns: {list(dummies.columns)}")

    print("\n[5] Select which method to use")
    print("Label   - Use for categories with a natural order, e.g., small/medium/large, or for the target variable")
    print("One-Hot - Use for categories without a natural order, e.g., colors, provinces, ethnicities")
    print("Be careful: If there are many categories (e.g., 1000 provinces), One-Hot will create 1000 columns")

    print("\n[6] Plotting Graphs")
    plot_counts(counts.values, counts.index.tolist(), OUTPUT_DIR)
    plot_onehot(encoded, labels, OUTPUT_DIR)


if __name__ == "__main__":
    main()
