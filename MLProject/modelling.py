import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # Ganti ke Regressor
from sklearn.metrics import mean_squared_error  # Ganti metrik ke regresi
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Memuat dataset
dataset_path = "MLProject/namadataset_preprocessing/Mental_Health_and_Social_Media_Balance_No_Outlier.csv"
df = pd.read_csv(dataset_path)

# Label Encoding untuk kolom 'Gender' dan kolom lainnya yang kategorikal
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])

# One-Hot Encoding untuk kolom 'Social_Media_Platform' jika ada lebih dari dua kategori
df = pd.get_dummies(df, columns=['Social_Media_Platform'])

# Memisahkan fitur dan label
X = df.drop(columns=['Happiness_Index(1-10)', 'User_ID'])  # Sesuaikan dengan kolom yang ingin Anda ambil sebagai fitur
y = df['Happiness_Index(1-10)']  # Kolom label (target) yang kontinu

# Split data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Melatih model RandomForestRegressor (ganti dari Classifier ke Regressor)
model = RandomForestRegressor(n_estimators=100, random_state=42)

# MLflow Tracking UI: Start logging
mlflow.start_run()

# Melatih model
model.fit(X_train, y_train)

# Prediksi dan evaluasi model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)  # Menggunakan Mean Squared Error untuk regresi

# Log model dan metrik ke MLflow
mlflow.log_param("model_type", "RandomForestRegressor")
mlflow.log_param("n_estimators", 100)
mlflow.log_metric("mse", mse)

# Menyimpan model ke MLflow
mlflow.sklearn.log_model(model, "model")

# Menyelesaikan run
mlflow.end_run()

print(f"Model Training Complete. Mean Squared Error: {mse}")
