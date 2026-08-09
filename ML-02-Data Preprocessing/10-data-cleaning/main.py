



import os
import numpy as np
from load_data import load_data, OUTPUT_DIR
from plot_result import plot_steps, plot_before_after


def main():
    print("-" * 30)
    print("10 - Data Cleaning")
    print("-" * 30)

    df = load_data()
    before = df["age"].to_numpy()
    steps = {"original": len(df)}

    print(f"\nstart {len(df)} rows")

    print("\n[1] name columns consistently")
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    print(f"Columns: {list(df.columns)}")

    print("\n[2] check data types")
    print(df.dtypes.to_string())

    print("\n[3] remove duplicate rows")
    df = df.drop_duplicates(subset="img_name")
    steps["remove duplicates"] = len(df)
    print(f"Remaining: {len(df)} rows")

    print("\n[4] remove rows with missing values")
    df = df.dropna()
    steps["remove missing values"] = len(df)
    print(f"Remaining: {len(df)} rows")

    print("\n[5] remove invalid values")
    # age must be between 1-120, gender must be 0 or 1
    df = df[(df["age"] >= 1) & (df["age"] <= 120)]
    df = df[df["gender"].isin([0, 1])]
    steps["remove invalid values"] = len(df)
    print(f"Remaining: {len(df)} rows")

    print("\n[6] remove outliers with IQR")
    q1, q3 = df["age"].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    df = df[(df["age"] >= low) & (df["age"] <= high)]
    steps["remove outliers"] = len(df)
    print(f"Bounds {low:.1f} to {high:.1f}, remaining: {len(df)} rows")

    print("\n[7] reset index")
    df = df.reset_index(drop=True)

    print("\n[8] save cleaned file")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "cleaned.csv")
    df.to_csv(path, index=False)
    print(f"Saved: cleaned.csv ({len(df)} rows)")

    print("\n[9] summary")
    removed = steps["original"] - len(df)
    print(f"Removed a total of {removed} rows "
          f"({removed / steps['original'] * 100:.1f}%)")
    print(f"Average age before: {np.mean(before):.2f} -> after: {df['age'].mean():.2f}")

    print("\n[10] plot graphs")
    plot_steps(steps, OUTPUT_DIR)
    plot_before_after(before, df["age"].to_numpy(), OUTPUT_DIR)


if __name__ == "__main__":
    main()
