import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.db import DATA_RAW, MODELS_DIR
from src.predict import CarPricePredictor
from src.features import get_feature_lists

def main():
    print("Загрузка данных и моделей...")
    df = pd.read_parquet(DATA_RAW / "cars.parquet")
    predictor = CarPricePredictor(MODELS_DIR)
    
    df_test = df.sample(frac=0.2, random_state=42).copy()
    print(f"Тестовая выборка: {len(df_test)} машин\n")
    
    results = []
    features, _, _, cat_f = get_feature_lists(df)
    
    X = df_test[features].copy()
    
    # 1. Предсказания квантилей
    q10_log = predictor.q10.predict(X) - predictor.q_hat
    q50_log = predictor.q50.predict(X)
    q90_log = predictor.q90.predict(X) + predictor.q_hat
    
    # Фикс crossing
    q10_log = np.minimum(q10_log, q50_log)
    q90_log = np.maximum(q90_log, q50_log)
    
    # 2. Предсказание уверенности (ЧИСТЫЙ ML, без эвристик)
    X_conf = X.copy()
    X_conf["q10_log"] = q10_log
    X_conf["q50_log"] = q50_log
    X_conf["q90_log"] = q90_log
    X_conf["width_log"] = q90_log - q10_log
    
    raw_confidence = predictor.conf_model.predict_proba(X_conf)[:, 1]
    
    # 3. Считаем покрытие
    for i in range(len(X)):
        low = np.expm1(q10_log[i])
        high = np.expm1(q90_log[i])
        covered = int(low <= df_test.iloc[i]["price"] <= high)
        
        results.append({
            "confidence": raw_confidence[i],
            "covered": covered
        })
        
    res_df = pd.DataFrame(results)
    
    # 4. Анализ калибровки
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    res_df["bucket"] = pd.cut(res_df["confidence"], bins=bins, labels=labels, include_lowest=True)
    
    calib = res_df.groupby("bucket", observed=False).agg(
        n=("covered", "size"),
        avg_confidence=("confidence", "mean"),
        actual_coverage=("covered", "mean")
    ).reset_index()
    
    calib["avg_confidence"] = (calib["avg_confidence"] * 100).round(1)
    calib["actual_coverage"] = (calib["actual_coverage"] * 100).round(1)
    calib["diff"] = (calib["actual_coverage"] - calib["avg_confidence"]).round(1)
    
    print("─── Калибровка уверенности ───")
    print(calib.to_string(index=False))
    
    # 5. График калибровки
    plt.figure(figsize=(8, 6))
    plt.plot([0, 100], [0, 100], "k--", label="Идеальная калибровка")
    
    valid = calib.dropna(subset=["actual_coverage"])
    plt.plot(valid["avg_confidence"], valid["actual_coverage"], "o-", color="red", label="Наша модель")
    
    plt.xlabel("Предсказанная уверенность (%)")
    plt.ylabel("Фактическое покрытие (%)")
    plt.title("Калибровочная кривая (Reliability Diagram)")
    plt.legend()
    plt.grid(True)
    
    report_path = PROJECT_ROOT / "reports" / "confidence_calibration.png"
    report_path.parent.mkdir(exist_ok=True)
    plt.savefig(report_path)
    print(f"\nГрафик сохранен: {report_path}")
    plt.show()

if __name__ == "__main__":
    main()