


import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from load_data import load_data, OUTPUT_DIR
from plot_result import plot_transforms, plot_skew


def main():
    print("-" * 30)
    print("06 - Data Transformation")
    print("-" * 30)

    df = load_data()
    age = df[["age"]]

    print("\n[1] show the shape of the original data")
    skew = df["age"].skew()
    print(f"skewness = {skew:.3f}")
    print("Right-skewed (skew > 0) means the tail is long on the right side, indicating many people are young.")

    print("\n[2] transform with different methods")
    results = {"Original": df["age"].to_numpy(dtype=float)}

    results["log1p"] = np.log1p(df["age"])
    results["sqrt"] = np.sqrt(df["age"])
    results["Yeo-Johnson"] = PowerTransformer().fit_transform(age).ravel()
    results["Quantile"] = QuantileTransformer(
        output_distribution="normal", n_quantiles=1000, random_state=42
    ).fit_transform(age).ravel()

    print(f"\n{'Method':<16}{'Skewness':>10}{'Min':>9}{'Max':>9}")
    skews = []
    for name, values in results.items():
        s = pd.Series(values).skew()
        skews.append(s)
        print(f"{name:<16}{s:>10.3f}{np.min(values):>9.2f}{np.max(values):>9.2f}")

    print("\n[3] why transform the data")
    print("Linear models perform better when the data is normally distributed")
    print("Transformation helps pull the long tail in, reducing the influence of extreme values")

    print("\n[4] things to be careful about")
    print("The transformed results will be in a different scale from the original data, so you need to transform them back before interpretation")
    print("log ใช้กับค่าติดลบไม่ได้ ถ้ามีค่าติดลบให้ใช้ Yeo-Johnson")

    print("\n[5] plot the results")
    plot_transforms(results, OUTPUT_DIR)
    plot_skew(list(results), skews, OUTPUT_DIR)

if __name__ == "__main__":
    main()
