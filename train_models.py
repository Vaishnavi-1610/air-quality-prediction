import os
import shutil
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Note: CatBoost is not installed. To evaluate CatBoost, run: pip install catboost")

# --------------------------------------------------
# 1. Load the dataset
# --------------------------------------------------

if os.path.exists("AirQualityUCI.xlsx"):
    df = pd.read_excel("AirQualityUCI.xlsx")
elif os.path.exists("AirQualityUCI.csv"):
    df = pd.read_csv(
        "AirQualityUCI.csv",
        sep=";",
        decimal=",",
        engine="python"
    )
    if len(df.columns) == 1:
        df = pd.read_csv("AirQualityUCI.csv", sep=None, engine="python")
else:
    raise FileNotFoundError(
        "Place AirQualityUCI.xlsx or AirQualityUCI.csv "
        "in the same folder as train_models.py"
    )

# Clean column names
df.columns = df.columns.astype(str).str.strip()

# Remove completely empty columns and rows
df = df.dropna(axis=1, how="all")
df = df.dropna(axis=0, how="all")

# --------------------------------------------------
# 2. Prepare date and time
# --------------------------------------------------

if "Date" in df.columns and "Time" in df.columns:
    time_values = df["Time"].astype(str).str.strip()
    time_values = time_values.str.replace(".", ":", regex=False)

    df["DateTime"] = pd.to_datetime(
        df["Date"].astype(str).str.strip() + " " + time_values,
        dayfirst=True,
        errors="coerce"
    )

    df = df.sort_values("DateTime", kind="stable")
    df = df.reset_index(drop=True)

    df["hour"] = df["DateTime"].dt.hour
    df["day"] = df["DateTime"].dt.day
    df["month"] = df["DateTime"].dt.month

    df = df.drop(columns=["Date", "Time", "DateTime"])

# --------------------------------------------------
# 3. Clean numeric values
# --------------------------------------------------

for column in df.columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# In this dataset, -200 represents a missing measurement
df = df.replace(-200, np.nan)

target = "NO2(GT)"

if target not in df.columns:
    raise ValueError(
        f"Target column '{target}' was not found.\n"
        f"Available columns: {df.columns.tolist()}"
    )

# --------------------------------------------------
# 4. Predict the NEXT NO2 observation
# --------------------------------------------------

# The target is shifted by one row.
# Inputs come from the current observation;
# the answer comes from the following observation.

df["NO2_next"] = df[target].shift(-1)

# Current NO2 is allowed as an input because it is
# known when predicting the next observation.
df = df.rename(columns={target: "NO2_current"})

feature_columns = [
    column for column in df.columns
    if column not in ["NO2_next"]
]

X = df[feature_columns].copy()
y = df["NO2_next"].copy()

# The final row has no following observation.
valid = y.notna()
X = X.loc[valid].reset_index(drop=True)
y = y.loc[valid].reset_index(drop=True)

# Remove columns that contain no usable measurements
feature_columns = [
    column for column in X.columns
    if X[column].notna().any()
]
X = X[feature_columns]

if len(X) < 20:
    raise ValueError(
        "Too few usable observations. Check the dataset format "
        "and the NO2(GT) column."
    )

# --------------------------------------------------
# 5. Chronological train/test split (No shuffling)
# --------------------------------------------------

split_index = int(len(X) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nDataset loaded successfully!")
print("Total observations:", len(X))
print("Training observations (80% chronological):", len(X_train))
print("Testing observations (20% holdout):", len(X_test))
print("Number of input features:", len(feature_columns))

# --------------------------------------------------
# 6. Create the ML candidate models
# --------------------------------------------------

models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}

if CATBOOST_AVAILABLE:
    models["CatBoost"] = CatBoostRegressor(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=0
    )

results = []
trained_models = {}

# --------------------------------------------------
# 7. Train and evaluate all candidate models
# --------------------------------------------------

for name, estimator in models.items():
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", estimator)
    ])

    print(f"\nTraining {name}...")

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    trained_models[name] = pipeline

# --------------------------------------------------
# 8. Compare model performance
# --------------------------------------------------

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("MAE").reset_index(drop=True)

print("\n========== MODEL COMPARISON (HOLDOUT TEST) ==========")
print(results_df.to_string(index=False, float_format="%.4f"))

# Lowest MAE wins; RMSE and R2 are also reported
best_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]

print("\nChampion model based on test MAE:", best_name)

# --------------------------------------------------
# 9. Save CatBoost separately and preserve backups
# --------------------------------------------------

# Backup previous best model if it exists
if os.path.exists("air_quality_best_model.joblib"):
    shutil.copy("air_quality_best_model.joblib", "air_quality_best_model_backup.joblib")
    print("Created backup of previous model: air_quality_best_model_backup.joblib")

# Save CatBoost model bundle separately if trained
if "CatBoost" in trained_models:
    cb_output = {
        "model": trained_models["CatBoost"],
        "model_name": "CatBoost",
        "features": feature_columns,
        "target": "NO2_next",
        "prediction_description": (
            "CatBoost regressor predicting the next recorded NO2(GT) observation "
            "using chronological multi-sensor inputs."
        )
    }
    joblib.dump(cb_output, "air_quality_catboost.joblib")
    print("Saved dedicated CatBoost bundle: air_quality_catboost.joblib")

# Save champion model bundle
champion_output = {
    "model": best_model,
    "model_name": best_name,
    "features": feature_columns,
    "target": "NO2_next",
    "prediction_description": (
        f"{best_name} champion regressor predicting the next recorded NO2(GT) observation "
        "using chronological multi-sensor inputs."
    )
}
joblib.dump(champion_output, "air_quality_best_model.joblib")
print(f"Saved champion model bundle ({best_name}): air_quality_best_model.joblib")

# Save updated comparison CSV
results_df.to_csv("model_comparison.csv", index=False)
print("Saved comparison table: model_comparison.csv")
print("Existing air_quality_models.joblib (K-Means) was preserved untouched.")
