



import numpy as np
from load_data import load_data, OUTPUT_DIR
from plot_result import plot_boxplot, plot_outliers


def main():
    print("=" * 55)
    print("02 - Duplicate Data, Noise, and Outliers")
    print("=" * 55)

    df = load_data()

    print("\n[1] Duplicate Data")
    print(f"Rows with all columns duplicated  : {df.duplicated().sum()}")
    print(f"Duplicate filenames (img_name): {df.duplicated(subset='img_name').sum()}")

    clean = df.drop_duplicates(subset="img_name")
    print(f"After removing duplicates: {len(clean)} out of {len(df)} rows")

    print("\n[2] Outliers - IQR Method")
    q1, q3 = clean["age"].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    print(f"Q1 = {q1:.1f}, Q3 = {q3:.1f}, IQR = {iqr:.1f}")
    print(f"Normal range = {low:.1f} to {high:.1f}")

    outliers = clean[(clean["age"] < low) | (clean["age"] > high)]
    print(f"Found {len(outliers)} outliers ({len(outliers) / len(clean) * 100:.1f}%)")
    print(f"Ages considered outliers: {sorted(int(a) for a in outliers['age'].unique())[:10]} ...")

    print("\n[3] Outliers - Z-score Method")
    z = (clean["age"] - clean["age"].mean()) / clean["age"].std()
    print(f"|z| > 3: {(z.abs() > 3).sum()} rows")

    print("\n[4] Noise - Random small errors added to data")
    # noise คือความคลาดเคลื่อนเล็กๆ ที่ปนมากับข้อมูล ลองใส่ดู
    rng = np.random.default_rng(42)
    noisy = clean["age"] + rng.normal(0, 3, len(clean))
    print(f"Mean before adding noise: {clean['age'].mean():.2f}")
    print(f"Mean after adding noise: {noisy.mean():.2f}  (mean almost unchanged)")
    print(f"SD before: {clean['age'].std():.2f}  SD after: {noisy.std():.2f}  (SD increased)")

    print("\n[5] Decision Making")
    kept = clean[(clean["age"] >= low) & (clean["age"] <= high)]
    print(f"If we remove outliers, we will be left with {len(kept)} rows")
    print("However, ages 80-116 are realistic values, so we should not remove them")

    print("\n[6] Plotting Graphs")
    plot_boxplot(clean["age"], OUTPUT_DIR)
    plot_outliers(clean["age"], low, high, OUTPUT_DIR)


if __name__ == "__main__":
    main()
