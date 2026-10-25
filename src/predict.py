import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, CatBoostClassifier
from src.features import build_features


class CarPricePredictor:
    def __init__(self, models_dir):
        models_dir = Path(models_dir)

        self.features = json.loads((models_dir / "features.json").read_text())
        self.cat_features = json.loads((models_dir / "cat_features.json").read_text())
        self.q_hat = float((models_dir / "q_hat.txt").read_text().strip())

        self.q10 = CatBoostRegressor().load_model(str(models_dir / "catboost_final_q10.cbm"))
        self.q50 = CatBoostRegressor().load_model(str(models_dir / "catboost_final_q50.cbm"))
        self.q90 = CatBoostRegressor().load_model(str(models_dir / "catboost_final_q90.cbm"))
        self.conf_model = CatBoostClassifier().load_model(str(models_dir / "confidence.cbm"))

    def predict(self, car_dict: dict) -> dict:
        df = pd.DataFrame([car_dict])
        df = build_features(df, verbose=False)

        for f in self.features:
            if f not in df.columns:
                df[f] = "Unknown" if f in self.cat_features else False

        X = df[self.features]

        q10_log = self.q10.predict(X)[0] - self.q_hat
        q50_log = self.q50.predict(X)[0]
        q90_log = self.q90.predict(X)[0] + self.q_hat

        # Фикс quantile crossing
        q10_log = min(q10_log, q50_log)
        q90_log = max(q90_log, q50_log)

        # Расчет уверенности
        X_conf = X.copy()
        X_conf["q10_log"] = q10_log
        X_conf["q50_log"] = q50_log
        X_conf["q90_log"] = q90_log
        X_conf["width_log"] = q90_log - q10_log
        
        confidence = float(self.conf_model.predict_proba(X_conf)[0, 1])

        return {
            "price": price,
            "range_80": (low, high),
            "confidence": confidence,
        }