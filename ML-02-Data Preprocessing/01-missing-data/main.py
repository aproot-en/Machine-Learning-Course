



import numpy as np
from sklearn.impute import SimpleImputer

from load_data import load_data, OUTPUT_DIR
from plot_result import plot_missing, plot_compare


def main():
    print("=" * 55)
    print("01 - Missing Data")
    print("=" * 55)

    df = load_data()

    # Randomly remove 10% of the age values to create missing data
    rng = np.random.default_rng(42)
    holes = rng.choice(len(df), size=int(len(df) * 0.1), replace=False)
    df.loc[holes, "age"] = np.nan

    print("\n[1] Count missing values")
    print(df.isna().sum())
    print(f"\nProportion of missing values in age: {df['age'].isna().mean() * 100:.1f}%")

    print("\n[2] Method 1 - Drop rows with missing values.")
    dropped = df.dropna()
    print(f"เหลือ {len(dropped)} จาก {len(df)} แถว (หายไป {len(df) - len(dropped)})")

    print("\n[3] Method 2 - Fill with median (fillna)")
    filled = df.copy()
    filled["age"] = filled["age"].fillna(df["age"].median())
    print(f"เติมด้วย median = {df['age'].median():.1f}")

    print("\n[4] Method 3 - Use SimpleImputer from sklearn")
    imputer = SimpleImputer(strategy="mean")
    imputed = imputer.fit_transform(df[["age"]])
    print(f"เติมด้วย mean = {imputer.statistics_[0]:.1f}")

    print("\n[5] Compare methods")
    print(f"{'Method':<22}{'Number of Rows':>10}{'Mean':>12}")
    print(f"{'Original (with NaN)':<22}{len(df):>10}{df['age'].mean():>12.2f}")
    print(f"{'dropna':<22}{len(dropped):>10}{dropped['age'].mean():>12.2f}")
    print(f"{'fillna(median)':<22}{len(filled):>10}{filled['age'].mean():>12.2f}")
    print(f"{'SimpleImputer(mean)':<22}{len(imputed):>10}{imputed.mean():>12.2f}")

    print("\n[6] วาดกราฟ")
    plot_missing(df, OUTPUT_DIR)
    plot_compare(df["age"], dropped["age"], filled["age"], OUTPUT_DIR)


if __name__ == "__main__":
    main()
