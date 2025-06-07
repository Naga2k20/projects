import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# ✅ Load Data
csv_file = "balanced_cleaned_dataset.csv"
df = pd.read_csv(csv_file)

# ✅ Compute Derived Features
df["Study_Efficiency"] = df["Daily_Study_Hours"] * df["Attendance"]
df["Exam_Proficiency"] = (df["HSC_Score"] + df["SSC_Score"]) / 2
df["English_Impact"] = df["English_Proficiency"] * 0.78

# ✅ Define Features & Target
features = [
    "HSC_Score", "SSC_Score", "Attendance", "English_Proficiency", "Daily_Study_Hours",
    "Study_Efficiency", "Exam_Proficiency", "English_Impact"
]
target = "GPA"

# ✅ Check Feature Count
if not set(features).issubset(df.columns):
    missing = list(set(features) - set(df.columns))
    raise KeyError(f"❌ Missing features in dataset: {missing}")

# ✅ Split Data
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Print Min/Max Values Before Scaling (Debugging)
print("\n📊 Feature Min Values (Before Scaling):\n", X_train.min())
print("\n📊 Feature Max Values (Before Scaling):\n", X_train.max())

# ✅ Scale Data
scaler_path = "scaler.pkl"
try:
    scaler = joblib.load(scaler_path)  # Load existing scaler if available
    print("✅ Using existing scaler.")
except FileNotFoundError:
    print("⚠️ Scaler not found. Creating a new one.")
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    joblib.dump(scaler, scaler_path)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ✅ Print Min/Max Values After Scaling (Debugging)
print("\n📊 Feature Min Values (After Scaling):\n", pd.DataFrame(X_train_scaled, columns=X_train.columns).min())
print("\n📊 Feature Max Values (After Scaling):\n", pd.DataFrame(X_train_scaled, columns=X_train.columns).max())

# ✅ Check unique values for sanity
print("📊 Unique Values in Features:\n", X_train.nunique())

# ✅ Train XGBoost with More Trees & Depth
model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=0.2,
    reg_lambda=1.0,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# Get predictions from the training data
train_predictions = model.predict(X_train_scaled)

# Print the performance metrics for training data
print("Training MAE:", mean_absolute_error(y_train, train_predictions))
print("Training MSE:", mean_squared_error(y_train, train_predictions))
print("Training R²:", r2_score(y_train, train_predictions))

# ✅ Evaluate Model on Test Data
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Performance:")
print(f"🔹 Mean Absolute Error (MAE): {mae:.4f}")
print(f"🔹 Mean Squared Error (MSE): {mse:.4f}")
print(f"🔹 R² Score: {r2:.4f}")

# Display sample predictions from the test set
sample_test = X_test_scaled[:5]
sample_preds = model.predict(sample_test)
print("📌 Sample Predictions from Test Data:", sample_preds)

# ✅ Save Model
model_path = "student_gpa_model.pkl"
joblib.dump(model, model_path)
print(f"✅ Model training completed! Saved as '{model_path}'")

# ✅ Verify Feature Count in Model
loaded_model = joblib.load(model_path)
if loaded_model.n_features_in_ != len(features):
    raise ValueError(f"❌ Feature shape mismatch! Model expects {loaded_model.n_features_in_} features but dataset has {len(features)}.")
else:
    print(f"✅ Model trained successfully with {loaded_model.n_features_in_} features.")
