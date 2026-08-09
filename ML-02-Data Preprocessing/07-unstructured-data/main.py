


import numpy as np
from load_data import load_data, load_pixels, OUTPUT_DIR
from plot_result import plot_faces, plot_pixel_hist


def main():
    print("-" * 30)
    print("07 - Unstructured Data")
    print("-" * 30)

    print("\n[1] the structured data (Structured) - located in a table")
    df = load_data()
    print(df.head(3))
    print("Each column has a clear meaning and can be used directly with models")

    print("\n[2] the unstructured data (Unstructured) - the 'pixels' column")
    images, meta = load_pixels(200)

    print(f"1 column = images 48x48 = {48 * 48} numbers per row")
    print(f"kept as a string in the file: '129 128 128 126 ...'")
    print("must split and convert to numbers before use")

    print("\n[3] image statistics")
    print(f"shape       : {images.shape}")
    print(f"dtype       : {images.dtype}")
    print(f"minimum     : {images.min()}")
    print(f"maximum     : {images.max()}")
    print(f"mean        : {images.mean():.1f}")

    print("\n[4] transform for model compatibility")

    #  0-1 
    normalized = images.astype(np.float32) / 255.0
    print(f"normalize   : {images.min()}-{images.max()} -> "
          f"{normalized.min():.1f}-{normalized.max():.1f}")

    flat = normalized.reshape(len(normalized), -1)
    print(f"flatten     : {images.shape} -> {flat.shape}")

    print("\n[5] Summary: structured vs unstructured data")
    print("Unstructured data must be transformed into numerical values before use")
    print("  Images  -> Brightness matrices")
    print("  Text    -> Bag of Words / TF-IDF / embedding")
    print("  Audio   -> Sound waves / spectrograms")

    print("\n[6] plot the results")
    plot_faces(images, meta, OUTPUT_DIR)
    plot_pixel_hist(images, OUTPUT_DIR)


if __name__ == "__main__":
    main()

