# sanity_check.py
import json, pandas as pd, numpy as np, bentoml

CSV_PATH = "ML_modele_iqr_cat.csv"
TARGET = "SiteEnergyUse(kBtu)"
FEATURES = [
    "BuildingAge","Surface moyenne par étage","Ratio Largest Use / Total",
    "NumberofBuildings","NumberofFloors","PropertyGFATotal","PropertyGFAParking",
    "PropertyGFABuilding(s)","LargestPropertyUseTypeGFA","SecondLargestPropertyUseTypeGFA",
    "PropertyCategory",
]

# 1) repères
df = pd.read_csv(CSV_PATH)
q = df[TARGET].quantile([0.05,0.25,0.5,0.75,0.95]).to_dict()
print("[Target quantiles kBtu]", {int(k*100): float(v) for k,v in q.items()})

# 2) modèle
ref = bentoml.models.get("energy_model:latest")
model = bentoml.sklearn.load_model(ref)
md = ref.info.metadata or {}
use_log = bool(md.get("use_log_target", False))

# 3) payload
with open("request.json","r",encoding="utf-8") as f:
    p = json.load(f)

# reconstruire la catégorie si one-hot
if "PropertyCategory" not in p:
    cat_cols = [c for c in p if c.startswith("PropertyCategory_")]
    if not cat_cols:
        raise SystemExit("Aucune colonne PropertyCategory_*.")
    best = max(cat_cols, key=lambda c: float(p.get(c,0) or 0))
    p["PropertyCategory"] = best.split("PropertyCategory_",1)[1]

X = pd.DataFrame([p], columns=FEATURES)
y = model.predict(X)
if use_log: y = np.expm1(y)

print("[Request used]", X.iloc[0].to_dict())
print("[Prediction kBtu]", float(np.ravel(y)[0]))
