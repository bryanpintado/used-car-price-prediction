from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocess import clean_data

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "train.csv"
MODEL_PATH = BASE_DIR / "data" / "processed" / "random_forest_model.joblib"
RESULTS_PATH = BASE_DIR / "data" / "processed" / "final_model_results.csv"

df = pd.read_csv(RAW_DATA_PATH)
df = clean_data(df)

X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = joblib.load(MODEL_PATH)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

results = pd.DataFrame({
    "Model": ["Random Forest"],
    "MAE": [mae],
    "RMSE": [rmse],
    "R2 Score": [r2]
})

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(RESULTS_PATH, index=False)

print(results)
print(f"Results saved to {RESULTS_PATH}")