import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocess import clean_data

df = pd.read_csv("data/raw/train.csv")
df = clean_data(df)

X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = joblib.load("data/processed/random_forest_model.joblib")

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

results.to_csv("data/processed/final_model_results.csv", index=False)

print(results)
print("Evaluation results saved.")