





## Read data from CSV file and prepare for clustering mini-project 

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

CSV_PATH = Path(__file__).resolve().parent.parent / "data-animal" / "animal_dataset.csv"

# feature ที่ใช้จัดกลุ่ม (ตัวเลขล้วน)
FEATURES = [
    "Average_Lifespan (Years)",
    "Weight (kg)",
    "Height (cm)",
    "Speed (km/h)",
    "Endangered_Level (1-5)",
]


# ---------------------------------------------------------------------------
def load_data():
    """
    คืนค่าเป็น dict ที่มี
        X        : ข้อมูลหลัง scale แล้ว (ใช้จัดกลุ่ม)
        X_raw    : ข้อมูลหน่วยจริง (ใช้ตอนอธิบายผล เช่น "กลุ่มนี้หนักเฉลี่ย 1200 kg")
        df       : ตารางเต็มจากไฟล์ CSV
    """
    df = pd.read_csv(CSV_PATH)
    df = df.dropna()

    X_raw = df[FEATURES].to_numpy(dtype="float32")  
    X = StandardScaler().fit_transform(X_raw).astype("float32")  ## transform to mean=0, std=1 (scale) for clustering 

    return {"X": X, "X_raw": X_raw, "df": df, "features": FEATURES}


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data = load_data()
    print("size data :", data["X"].shape)
    print("mean after scale (should be close to 0) :", data["X"].mean(axis=0).round(3))
