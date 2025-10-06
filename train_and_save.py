# train_and_save.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import bentoml

# ====== 1) Données ======
CSV_PATH = "ML_modele_iqr_cat.csv"   # ou "ML_modele.csv"
df = pd.read_csv(CSV_PATH)

TARGET = "SiteEnergyUse(kBtu)"
FEATURES_NUM = [
    "BuildingAge",
    "Surface moyenne par étage",
    "Ratio Largest Use / Total",
    "NumberofBuildings",
    "NumberofFloors",
    "PropertyGFATotal",
    "PropertyGFAParking",
    "PropertyGFABuilding(s)",
    "LargestPropertyUseTypeGFA",
    "SecondLargestPropertyUseTypeGFA",
]
FEATURES_CAT = ["PropertyCategory"]
FEATURES = FEATURES_NUM + FEATURES_CAT

df = df.dropna(subset=[TARGET] + FEATURES)
X = df[FEATURES].copy()
y = df[TARGET].astype(float).copy()

# ====== 2) Option: log1p ======
USE_LOG_TARGET = True
if USE_LOG_TARGET:
    y = np.log1p(y)

# ====== 3) Pipeline ======
preprocess = ColumnTransformer(
    transformers=[
        ("num", "passthrough", FEATURES_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
    ],
    remainder="drop",
)

pipe = Pipeline([
    ("prep", preprocess),
    ("rf", RandomForestRegressor(n_estimators=400, n_jobs=-1, random_state=42)),
])

# ====== 4) Train + Eval ======
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
pipe.fit(X_tr, y_tr)
pred_te = pipe.predict(X_te)

if USE_LOG_TARGET:
    y_te_lin = np.expm1(y_te)
    pred_te_lin = np.expm1(pred_te)
    mae = mean_absolute_error(y_te_lin, pred_te_lin)
    rmse = mean_squared_error(y_te_lin, pred_te_lin, squared=False)
    r2 = r2_score(y_te_lin, pred_te_lin)
else:
    mae = mean_absolute_error(y_te, pred_te)
    rmse = mean_squared_error(y_te, pred_te, squared=False)
    r2 = r2_score(y_te, pred_te)

print(f"[EVAL] MAE={mae:.1f}  RMSE={rmse:.1f}  R2={r2:.3f}")

# ====== 5) Stats de clip (quantiles 1%-99%) pour stabiliser en prod ======
q_lo = X_tr[FEATURES_NUM].quantile(0.01).to_dict()
q_hi = X_tr[FEATURES_NUM].quantile(0.99).to_dict()

# ====== 6) Save BentoML ======
tag = bentoml.sklearn.save_model(
    "energy_model",
    pipe,
    # tout doit être JSON-serializable
    metadata={
        "use_log_target": USE_LOG_TARGET,
        "expected_features": FEATURES,
        "numeric_features": FEATURES_NUM,
        "clip_min": {k: float(v) for k, v in q_lo.items()},
        "clip_max": {k: float(v) for k, v in q_hi.items()},
    },
    custom_objects={"feature_names": FEATURES},  # info annexe
)
print("✅ Saved model tag:", tag)
