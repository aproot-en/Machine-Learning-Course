


import os
import pandas as pd
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "..", "age_gender.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


def load_data():
    df = pd.read_csv(CSV_PATH, usecols=["age", "ethnicity", "gender", "img_name"])
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df
