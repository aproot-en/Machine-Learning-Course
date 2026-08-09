


from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from load_data import load_data, OUTPUT_DIR
from plot_result import plot_scalers, plot_range


def main():
    print("-" * 30)
    print("03 - Data Scaling and Normalization")
    print("-" * 30)

    df = load_data()
    age = df[["age"]]

    print("\n[1] Real data")
    print(f"min={age.min().item()}  max={age.max().item()}  "
          f"mean={age.mean().item():.2f}  std={age.std().item():.2f}")

    print("\n[2] Try all 3 methods")
    scalers = {
        "MinMaxScaler": MinMaxScaler(),      # Compress to range 0-1
        "StandardScaler": StandardScaler(),  # Make mean=0, std=1
        "RobustScaler": RobustScaler(),      # Use median and IQR, robust to outliers
    }

    results = {"Original": age["age"].to_numpy()}

    for name, scaler in scalers.items():
        results[name] = scaler.fit_transform(age).ravel()

    print(f"\n{'Method':<18}{'min':>9}{'max':>9}{'mean':>9}{'std':>9}")
    for name, values in results.items():
        print(f"{name:<18}{values.min():>9.2f}{values.max():>9.2f}"
              f"{values.mean():>9.2f}{values.std():>9.2f}")

    print("\n[3] Select which method to use")
    print("MinMaxScaler   - Want a fixed range of 0-1, e.g., images, but sensitive to outliers")
    print("StandardScaler - Most commonly used, suitable for data with a normal distribution")
    print("RobustScaler   - Data has many outliers, as it uses median instead of mean")

    print("\n[4] Important Considerations")
    print("Fit only on the training set, then transform the test set")
    print("If you fit on the entire dataset before splitting, you will leak information from the test set into the training set (data leakage)")

    print("\n[5] Plotting Graphs")
    plot_scalers(results, OUTPUT_DIR)
    plot_range(results, OUTPUT_DIR)


if __name__ == "__main__":
    main()
