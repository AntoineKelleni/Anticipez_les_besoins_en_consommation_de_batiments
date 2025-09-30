# service.py — BentoML 1.4+ (new style) avec validation & calcul auto
from __future__ import annotations

import typing as t
import numpy as np
import pandas as pd
import bentoml
from bentoml.models import BentoModel


@bentoml.service(name="energy_api")
class EnergyAPI:
    """
    Entrée POST (BentoML 1.4) : JSON enveloppé -> { "payload": {...} } ou { "payload": [ {...}, ... ] }
    - Accepte:
        • BRUT : 'PropertyCategory' donnée
        • ONE-HOT : colonnes 'PropertyCategory_*' (catégorie inférée)
    - Calcule/valide automatiquement :
        • Surface moyenne par étage = PropertyGFABuilding(s) / NumberofFloors
        • Ratio Largest Use / Total = LargestPropertyUseTypeGFA / PropertyGFABuilding(s)
    - Réponses incluent éventuellement: 'adjustments', 'warnings', 'violations'
    """

    model_ref = BentoModel("energy_model:latest")

    def __init__(self) -> None:
        self.model = bentoml.sklearn.load_model(self.model_ref)
        md = self.model_ref.info.metadata or {}

        self.use_log = bool(md.get("use_log_target", False))
        self.expected: list[str] = md.get(
            "expected_features",
            [
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
                "PropertyCategory",
            ],
        )
        self.numeric: list[str] = md.get(
            "numeric_features",
            [
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
            ],
        )
        self.clip_min: dict[str, float] = md.get("clip_min", {}) or {}
        self.clip_max: dict[str, float] = md.get("clip_max", {}) or {}

    # ---------------------------- helpers internes ---------------------------- #

    def _to_df(self, payload: t.Union[dict, list[dict]]) -> pd.DataFrame:
        return pd.DataFrame([payload]) if isinstance(payload, dict) else pd.DataFrame(payload)

    def _ensure_category(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, t.Optional[str], list[str], t.Optional[int]]:
        inferred_cat: t.Optional[str] = None
        ones_count: t.Optional[int] = None
        cat_cols = sorted([c for c in df.columns if c.startswith("PropertyCategory_")])

        if "PropertyCategory" not in df.columns and cat_cols:
            vals = df[cat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            idxmax = vals.values.argmax(axis=1) if not vals.empty else np.array([])
            inferred = []
            for i in range(len(df)):
                if vals.shape[1] == 0:
                    inferred.append("Other")
                    continue
                vmax = float(vals.iloc[i, idxmax[i]])
                inferred.append(
                    cat_cols[idxmax[i]].split("PropertyCategory_", 1)[1] if vmax > 0.0 else "Other"
                )
            df = df.copy()
            df["PropertyCategory"] = inferred
            inferred_cat = inferred[0] if len(inferred) else None
            if len(df) >= 1:
                ones_count = int((vals.iloc[0] > 0).sum())
        return df, inferred_cat, cat_cols, ones_count

    def _clip_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.clip_min or not self.clip_max:
            return df
        df = df.copy()
        for c in self.numeric:
            if c in df.columns:
                lo = self.clip_min.get(c, None)
                hi = self.clip_max.get(c, None)
                if lo is not None:
                    df[c] = np.maximum(pd.to_numeric(df[c], errors="coerce"), lo)
                if hi is not None:
                    df[c] = np.minimum(pd.to_numeric(df[c], errors="coerce"), hi)
        return df

    def _reconcile_derived(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict], list[str], list[str]]:
        """
        Calcule/valide :
          A) Surface moyenne par étage = PropertyGFABuilding(s) / NumberofFloors
          B) Ratio Largest Use / Total = LargestPropertyUseTypeGFA / PropertyGFABuilding(s)
        Renvoie (df_corrigé, adjustments, warnings, violations)
        """
        df = df.copy()
        adjustments: list[dict] = []
        warnings: list[str] = []
        violations: list[str] = []

        # noms courts
        s_floor = "Surface moyenne par étage"
        ratio = "Ratio Largest Use / Total"
        gfa_b = "PropertyGFABuilding(s)"
        floors = "NumberofFloors"
        gfa_largest = "LargestPropertyUseTypeGFA"
        gfa_second = "SecondLargestPropertyUseTypeGFA"

        # une seule ligne ou batch -> on itère proprement
        for i in range(len(df)):
            row = df.iloc[i]

            # Convertir num
            def num(v): return float(pd.to_numeric(v, errors="coerce")) if pd.notna(v) else np.nan

            # ---- Validations de base
            nb_floors = num(row.get(floors))
            gfa_build = num(row.get(gfa_b))
            gfa_l = num(row.get(gfa_largest))
            gfa_s = num(row.get(gfa_second))

            if pd.isna(nb_floors) or nb_floors <= 0:
                violations.append(f"[row {i}] {floors} must be > 0 (got {row.get(floors)!r})")
                continue
            if pd.isna(gfa_build) or gfa_build <= 0:
                violations.append(f"[row {i}] {gfa_b} must be > 0 (got {row.get(gfa_b)!r})")
                continue

            # ---- A) Surface moyenne par étage
            computed_avg = gfa_build / nb_floors
            prev = row.get(s_floor, None)
            if pd.isna(prev) or prev is None:
                df.at[i, s_floor] = computed_avg
                adjustments.append({"row": i, "field": s_floor, "action": "computed", "value": computed_avg})
            else:
                prev_num = num(prev)
                # tolérance relative 1%
                if not pd.isna(prev_num):
                    if abs(prev_num - computed_avg) > max(0.01 * computed_avg, 1e-6):
                        df.at[i, s_floor] = computed_avg
                        adjustments.append(
                            {"row": i, "field": s_floor, "action": "corrected", "from": prev_num, "to": computed_avg}
                        )
                else:
                    df.at[i, s_floor] = computed_avg
                    adjustments.append({"row": i, "field": s_floor, "action": "computed", "value": computed_avg})

            # ---- B) Ratio Largest Use / Total
            if pd.isna(gfa_l) or gfa_l < 0:
                warnings.append(f"[row {i}] {gfa_largest} missing/invalid; set to 0 for ratio.")
                gfa_l = 0.0
                df.at[i, gfa_largest] = 0.0

            if gfa_l > gfa_build:
                warnings.append(f"[row {i}] {gfa_largest} > {gfa_b} (capping to total).")
                gfa_l = gfa_build
                df.at[i, gfa_largest] = gfa_build

            computed_ratio = 0.0 if gfa_build == 0 else gfa_l / gfa_build
            computed_ratio = float(np.clip(computed_ratio, 0.0, 1.0))

            prev_r = row.get(ratio, None)
            if pd.isna(prev_r) or prev_r is None:
                df.at[i, ratio] = computed_ratio
                adjustments.append({"row": i, "field": ratio, "action": "computed", "value": computed_ratio})
            else:
                prev_rn = num(prev_r)
                if not pd.isna(prev_rn):
                    if abs(prev_rn - computed_ratio) > 0.02:  # tolérance 2 pts
                        df.at[i, ratio] = computed_ratio
                        adjustments.append(
                            {"row": i, "field": ratio, "action": "corrected", "from": prev_rn, "to": computed_ratio}
                        )
                else:
                    df.at[i, ratio] = computed_ratio
                    adjustments.append({"row": i, "field": ratio, "action": "computed", "value": computed_ratio})

            # Alerte douce si largest + second > total
            if not pd.isna(gfa_s) and gfa_l + gfa_s > gfa_build * 1.02:
                warnings.append(
                    f"[row {i}] {gfa_largest}+{gfa_second} > {gfa_b} (check areas)."
                )

        return df, adjustments, warnings, violations

    def _prepare(self, payload: t.Union[dict, list[dict]]) -> tuple[pd.DataFrame, dict]:
        df = self._to_df(payload)
        df, inferred_cat, cat_cols, ones_count = self._ensure_category(df)

        # calcule/valide les colonnes dérivées
        df, adjustments, warnings, violations = self._reconcile_derived(df)

        missing = [c for c in self.expected if c not in df.columns]
        extras = [c for c in df.columns if c not in self.expected and not c.startswith("PropertyCategory_")]

        # clip éventuel
        df_clipped = self._clip_numeric(df)

        info = {
            "missing": missing,
            "extras_not_used": extras,
            "cat_cols_present": cat_cols,
            "inferred_category": inferred_cat,
            "one_hot_positives": ones_count,
            "clipped": bool(self.clip_min and self.clip_max),
            "adjustments": adjustments,
            "warnings": warnings,
            "violations": violations,
        }
        return df_clipped, info

    # --------------------------------- API ---------------------------------- #

    @bentoml.api
    def inspect(self, payload: t.Union[dict, list[dict]]) -> dict:
        df, info = self._prepare(payload)
        used = None
        if not info["missing"]:
            used = df[self.expected].iloc[0].to_dict() if isinstance(payload, dict) else df[self.expected].to_dict(orient="records")
        return {"expected": self.expected, "used": used, **info}

    @bentoml.api
    def predict(self, payload: t.Union[dict, list[dict]]) -> dict:
        df, info = self._prepare(payload)
        if info["violations"]:
            # hard fail si conditions impossibles (ex: floors <= 0)
            return {"error": "invalid_input", **info}
        if info["missing"]:
            return {"error": "missing keys", **info}

        X = df[self.expected]
        y = self.model.predict(X)
        y = np.asarray(y).ravel()
        if self.use_log:
            y = np.expm1(y)

        out: dict = {"prediction_kBtu": float(y[0]) if y.size == 1 else [float(v) for v in y]}
        # On renvoie aussi les infos utiles de normalisation/ajustement
        if info["adjustments"] or info["warnings"]:
            out["adjustments"] = info["adjustments"]
            out["warnings"] = info["warnings"]
        return out

    @bentoml.api
    def health(self) -> dict:
        return {"status": "ok", "model_tag": str(self.model_ref.tag), "log_target": self.use_log}
