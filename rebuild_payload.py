import json, bentoml, unicodedata as ud

ref = bentoml.models.get("energy_model:latest")
features = list(ref.custom_objects.get("feature_names", []))

def default_value(name: str):
    n = name.lower()
    # dummies
    if name.startswith("PropertyCategory_"):
        return 1 if name == "PropertyCategory_Office" else 0
    # ratios
    if "ratio" in n or "fraction" in n or "share" in n:
        return 0.7
    # surfaces / gfa
    if any(k in n for k in ["gfa","area","surface","size","sqft","sf"]):
        if "parking" in n: return 30000
        if "second" in n: return 40000
        if "largest" in n: return 180000
        if "building(s" in n or "building_s" in n: return 220000
        return 250000
    # étages
    if "floor" in n or "floors" in n or "story" in n:
        return 5
    # âge / années
    if "age" in n or "année" in n or "an " in n or "year" in n or "years" in n:
        return 40
    # nombre de bâtiments
    if "numberofbuildings" in n or "nbuildings" in n or "buildings" in n:
        return 1
    # défaut
    return 100.0

# construit le payload EN UTILISANT EXACTEMENT les clés du modèle
payload = { name: default_value(name) for name in features }

with open("request.json","w",encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)

print("OK -> request.json aligné, nb clés:", len(payload))
