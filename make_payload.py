import json, bentoml

ref = bentoml.models.get("energy_model:latest")
features = list(ref.custom_objects.get("feature_names", []))

payload = {}
for f in features:
    lf = f.lower()

    # one-hot PropertyCategory_*
    if f.startswith("PropertyCategory_"):
        payload[f] = 1 if f == "PropertyCategory_Office" else 0
        continue

    # ratios
    if "ratio" in lf or "fraction" in lf or "share" in lf:
        payload[f] = 0.7
        continue

    # surfaces / GFA
    if any(k in lf for k in ["gfa","area","surface","size","sqft","sf"]):
        if "parking" in lf: payload[f] = 30000
        elif "second" in lf: payload[f] = 40000
        elif "largest" in lf: payload[f] = 180000
        elif "building(s" in lf or "building_s" in lf: payload[f] = 220000
        else: payload[f] = 250000
        continue

    # étages
    if "floor" in lf or "floors" in lf or "story" in lf:
        payload[f] = 5
        continue

    # âge / années
    if "âge" in f.lower() or "age" in lf or "year" in lf or "years" in lf:
        payload[f] = 40
        continue

    # nombre de bâtiments
    if "numberofbuildings" in lf or "nbuildings" in lf or "buildings" in lf:
        payload[f] = 1
        continue

    # défaut
    payload[f] = 100.0

# ASSURE la présence explicite de la clé accentuée
payload.setdefault("Surface moyenne par étage", 250000)

with open("request.json","w",encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)

print("OK -> request.json (", len(payload), "keys )")
