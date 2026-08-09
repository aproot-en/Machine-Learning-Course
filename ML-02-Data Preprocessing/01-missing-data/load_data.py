import os

import pandas as pd

# age_gender.csv is one level up
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "..", "age_gender.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


def load_data():
    """Read the 4 small columns. The 'pixels' column is skipped because it
    holds 2,304 numbers per row and would use several GB of RAM."""

    df = pd.read_csv(CSV_PATH, usecols=["age", "ethnicity", "gender", "img_name"])

    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    return df
