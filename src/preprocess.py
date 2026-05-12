from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "train.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_used_cars.csv"

def load_data(path=RAW_DATA_PATH):
    return pd.read_csv(path)

def clean_data(df):
    df = df.copy()

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    if "New_Price" in df.columns:
        df = df.drop(columns=["New_Price"])

    for col in ["Mileage", "Engine", "Power"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.extract(r"(\d+\.?\d*)").astype(float)

    return df

if __name__ == "__main__":
    df = load_data()
    cleaned_df = clean_data(df)

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"Cleaned data saved to {PROCESSED_DATA_PATH}")