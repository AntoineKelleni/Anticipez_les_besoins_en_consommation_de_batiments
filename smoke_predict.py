import json
import pandas as pd
import numpy as np
import bentoml

ref = bentoml.models.get("energy_model:latest")
model = bentoml.sklearn.load_model(ref)

with open("request.json", "r", encoding="utf-8") as f:
    payload = json.load(f)

X = pd.DataFrame([payload])
y = model.predict(X)
use_log = (ref.info.metadata or {}).get("use_log_target", False)
if use_log:
    y = np.expm1(y)

print("OK prediction:", float(np.ravel(y)[0]))
