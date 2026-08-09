


import os
import numpy as np
import pandas as pd
# age_gender.csv is one level up
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "..", "age_gender.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

def load_data():
    df = pd.read_csv(CSV_PATH, usecols=["age", "ethnicity", "gender", "img_name"])
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def load_pixels(n_rows=200):
    df = pd.read_csv(CSV_PATH, nrows=n_rows)
    images = np.array([np.array(s.split(), dtype=np.uint8) for s in df["pixels"]])
    images = images.reshape(-1, 48, 48)
    print(f"Loaded {len(images)} images, shape {images.shape}")

    return images, df[["age", "ethnicity", "gender"]]
