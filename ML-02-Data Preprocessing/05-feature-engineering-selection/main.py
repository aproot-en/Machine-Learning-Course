



import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

from load_data import load_data, OUTPUT_DIR
from plot_result import plot_new_feature, plot_scores


def main():
    print("-" * 30)
    print("05 - Feature Engineering and Feature Selection")
    print("-" * 30)

    df = load_data()
    print("\n[1] Feature Engineering - create new features from existing ones")

    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 12, 19, 35, 60, 120],
        labels=["child", "teen", "young", "adult", "senior"],
    )

    # แปลงเป็น 0/1
    df["is_adult"] = (df["age"] >= 18).astype(int)

    # ดึงปีจากชื่อไฟล์ 
    df["year"] = df["img_name"].str[:4].astype(int)

    print(" Create 3 groups: age_group, is_adult, year")
    print(df[["age", "age_group", "is_adult", "year"]].head())

    counts = df["age_group"].value_counts().sort_index()
    print(f"\nage_group:\n{counts}")
    print(f"\nPercentage of adults: {df['is_adult'].mean() * 100:.1f}%")

    print("\n[2] Feature Selection - select features that help with prediction")

    X = df[["age", "ethnicity", "is_adult", "year"]]
    y = df["gender"]

    selector = SelectKBest(score_func=f_classif, k=2)
    selector.fit(X, y)

    print("\npoint of feature importance:")
    for name, score in sorted(zip(X.columns, selector.scores_),
                              key=lambda p: -p[1]):
        print(f"  {name:<12}{score:>10.2f}")

    chosen = X.columns[selector.get_support()]
    print(f"\n The 2 best features are: {list(chosen)}")

    print("\n[3] Summary")
    print("Feature Engineering - adds data that makes the model easier to use")
    print("Feature Selection   - removes features that don't help, making the model faster and less prone to overfitting")

    print("\n[4] Plotting results")
    plot_new_feature(counts, OUTPUT_DIR)
    plot_scores(list(X.columns), selector.scores_, OUTPUT_DIR)

if __name__ == "__main__":
    main()
