import os
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# =========================================================
# FIX WAJIB: Paksa MLflow pakai path lokal (Linux-safe)
# =========================================================
os.makedirs("mlruns", exist_ok=True)
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("mental-health-ci")

# =========================================================
# Load dataset
# =========================================================
dataset_path = "MLProject/namadataset_preprocessing/Mental_Health_and_Social_Media_Balance_No_Outlier.csv"
df = pd.read_csv(dataset_path)

# =========================================================
# Encoding
# =========================================================
label_encoder = LabelEncoder()
df["Gender"] = label_encoder.fit_transform(df["Gender"])

df = pd.get_dummies(df, columns=["Social_Media_Platform"])

# =========================================================
# Split fitur & label
# =========================================================
X = df.drop(columns=["Happiness_Index(1-10)", "User_ID"])
y = df["Happiness_Index(1-10)"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================================================
# Model
# =========================================================
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# =========================================================
# MLflow Run
# =========================================================
with mlflow.start_run():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("mse", mse)

    mlflow.sklearn.log_model(model, "model")

print(f"Model Training Complete. Mean Squared Error: {mse}")