import json, bentoml
ref = bentoml.models.get("energy_model:latest")
expected = list(ref.custom_objects["feature_names"])
with open("request.json","r",encoding="utf-8") as f:
    got = list(json.load(f).keys())
print("== same length? ", len(expected)==len(got), len(expected))
print("Missing:", sorted(set(expected)-set(got)))
print("Extra  :", sorted(set(got)-set(expected)))
