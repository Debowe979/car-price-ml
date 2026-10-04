"""
Клиент для оценки одной машины из БД через API.

Использование:
    python clients/predict_from_db.py 575080427
    python clients/predict_from_db.py --batch 10   # случайные 10 машин
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import requests

# добавить корень проекта в sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.db import get_engine


API_URL = "http://localhost:8000/predict"

# поля, которые НЕ отправляем (утечки + служебные + таргет)
DROP_BEFORE_SEND = [
    # утечки
    "market_price", "deal_type", "price_discount", "views",
    "published_date", "json_ld", "bull_data",
    # служебные
    "id", "url", "source", "title", "list_date",
    "engine_raw", "vin_masked", "photos",
    "status", "detail_loaded", "parse_errors",
    "created_at", "updated_at", "modification_id",
    # таргет и JSONB-источники
    "price", "options", "condition_flags",
]


def row_to_car_dict(row: pd.Series) -> dict:
    """Превращает строку из БД в dict для API."""
    car = row.to_dict()

    # развернуть JSONB → opt_* и cond_*
    for jsonb_col, prefix in [("options", "opt_"), ("condition_flags", "cond_")]:
        val = car.get(jsonb_col)
        if isinstance(val, dict):
            for k, v in val.items():
                car[f"{prefix}{k}"] = bool(v)

    # выкинуть лишнее
    for k in DROP_BEFORE_SEND:
        car.pop(k, None)

    # NaN → None (JSON не понимает NaN)
    car = {k: (None if pd.isna(v) else v) for k, v in car.items()}

    # Timestamp → строка (если попадётся)
    for k, v in car.items():
        if hasattr(v, "isoformat"):
            car[k] = v.isoformat()

    return car


def fetch_car(car_id: int) -> dict:
    engine = get_engine()
    df = pd.read_sql(f"SELECT * FROM cars WHERE id = {car_id}", engine)
    if df.empty:
        raise ValueError(f"Машина id={car_id} не найдена")
    return row_to_car_dict(df.iloc[0])


def fetch_random(n: int = 10) -> pd.DataFrame:
    engine = get_engine()
    return pd.read_sql(f"SELECT * FROM cars ORDER BY RANDOM() LIMIT {n}", engine)


def call_api(car: dict, api_url: str) -> dict:
    resp = requests.post(api_url, json=car, timeout=30)
    resp.raise_for_status()
    return resp.json()

def format_result(car: dict, true_price: float | None, result: dict) -> str:
    brand = car.get("brand", "?")
    model = car.get("model", "?")
    year = car.get("year", "?")
    mileage = car.get("mileage_km")

    lines = [f"\n{brand} {model} {year}, пробег {mileage:,} км"]

    if true_price is not None:
        lines.append(f"  Реальная цена: {true_price:>14,.0f} ₽")

    lines.append(f"  Прогноз (q50): {result['price']:>14,.0f} ₽")
    lines.append(
        f"  Интервал 80%: [{result['range_80_low']:>12,.0f}, "
        f"{result['range_80_high']:>12,.0f}] ₽"
    )

    if true_price is not None:
        err_pct = abs(result["price"] - true_price) / true_price * 100
        covered = result["range_80_low"] <= true_price <= result["range_80_high"]
        lines.append(f"  Ошибка: {err_pct:.1f}%   Попал в интервал: {covered}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Оценка машины через API")
    parser.add_argument("car_id", nargs="?", type=int, help="ID машины из БД")
    parser.add_argument("--batch", type=int, help="Случайные N машин из БД")
    parser.add_argument("--api", default="http://localhost:8000/predict",
                        help="URL API")
    args = parser.parse_args()

    api_url = args.api
    health_url = api_url.replace("/predict", "/health")

    # health check
    try:
        health = requests.get(health_url, timeout=5).json()
        print(f"API: {health}")
    except Exception as e:
        print(f"API недоступен: {e}")
        print("Запусти: uvicorn src.api:app --host 127.0.0.1 --port 8000")
        sys.exit(1)

    if args.batch:
        print(f"\nОценка {args.batch} случайных машин из БД...")
        df = fetch_random(args.batch)
        for _, row in df.iterrows():
            true_price = row["price"]
            car = row_to_car_dict(row)
            try:
                result = call_api(car, api_url)
                print(format_result(car, true_price, result))
            except Exception as e:
                print(f"  Ошибка: {e}")
    elif args.car_id:
        car = fetch_car(args.car_id)
        result = call_api(car, api_url)
        engine = get_engine()
        true = pd.read_sql(
            f"SELECT price FROM cars WHERE id = {args.car_id}", engine
        ).iloc[0, 0]
        print(format_result(car, true, result))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()