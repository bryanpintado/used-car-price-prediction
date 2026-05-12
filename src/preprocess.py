import pandas as pd

def load_data(path):
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
    df = load_data("data/raw/train.csv")
    cleaned_df = clean_data(df)
    cleaned_df.to_csv("data/processed/cleaned_used_cars.csv", index=False)
    print("Cleaned data saved to data/processed/cleaned_used_cars.csv")