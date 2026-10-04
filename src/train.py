"""
Финальное обучение модели. Запускается через Task Scheduler.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import DATA_RAW, MODELS_DIR
from src.features import get_feature_lists


def main():
    df = pd.read_parquet(DATA_RAW / "cars.parquet")
    print(f"[load] {df.shape}")

    X_all = df.drop(columns=["price", "seller_description"], errors="ignore")
    y_all = np.log1p(df["price"])

    X_train, X_cal, y_train, y_cal = train_test_split(
        X_all, y_all, test_size=0.10, random_state=42
    )

    features, _, _, cat_f = get_feature_lists(df)
    features = [f for f in features if f in X_train.columns]
    cat_f = [f for f in cat_f if f in X_train.columns]

    print(f"[train] {len(X_train)} samples, {len(features)} features")

    ts = datetime.now().strftime("%Y%m%d")
    models = {}

    for q in [0.1, 0.5, 0.9]:
        print(f"\n[train] q={q}")
        m = CatBoostRegressor(
            loss_function=f"Quantile:alpha={q}",
            iterations=3000, learning_rate=0.05, depth=8,
            cat_features=cat_f, random_seed=42, verbose=500,
        )
        m.fit(X_train[features], y_train)
        # актуальный файл + архив с датой
        m.save_model(str(MODELS_DIR / f"catboost_final_q{int(q*100)}.cbm"))
        m.save_model(str(MODELS_DIR / f"catboost_final_q{int(q*100)}_{ts}.cbm"))
        models[q] = m

    # conformal
    q10_cal = models[0.1].predict(X_cal[features])
    q90_cal = models[0.9].predict(X_cal[features])
    scores = np.maximum(q10_cal - y_cal.values, y_cal.values - q90_cal)
    n = len(scores)
    level = min(np.ceil((n + 1) * 0.80) / n, 1.0)
    q_hat = float(np.quantile(scores, level))

    y_cal_rub = np.expm1(y_cal)
    q10_cp = np.expm1(q10_cal - q_hat)
    q90_cp = np.expm1(q90_cal + q_hat)
    cov = ((y_cal_rub >= q10_cp) & (y_cal_rub <= q90_cp)).mean()

    print(f"\n[conformal] q_hat = {q_hat:.5f}, coverage = {cov:.3f}")

    (MODELS_DIR / "q_hat.txt").write_text(str(q_hat))
    (MODELS_DIR / "features.json").write_text(json.dumps(features))
    (MODELS_DIR / "cat_features.json").write_text(json.dumps(cat_f))
    print(f"[save] metadata -> {MODELS_DIR}")


if __name__ == "__main__":
    main()