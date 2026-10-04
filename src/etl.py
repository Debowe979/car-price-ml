"""
ETL: PostgreSQL (drom.cars) -> data/raw/cars.parquet.

Читает таблицу, дропает утечки и служебные поля, разворачивает JSONB,
применяет features.build_features, сохраняет parquet.
"""
from pathlib import Path
import pandas as pd

from src.db import get_engine, DATA_RAW
from src.features import build_features


OUT_PATH = DATA_RAW / "cars.parquet"

DROP_COLUMNS = [
    # утечки
    "market_price", "deal_type", "price_discount", "views",
    "published_date", "json_ld", "bull_data",
    # служебное
    "id", "url", "source", "title", "list_date",
    "engine_raw", "vin_masked", "photos",
    "status", "detail_loaded", "parse_errors",
    "created_at", "updated_at", "modification_id",
]


def load_raw(engine) -> pd.DataFrame:
    df = pd.read_sql("SELECT * FROM cars", engine)
    print(f"[load] rows={len(df)}, cols={len(df.columns)}")
    return df


def drop_leaks(df: pd.DataFrame) -> pd.DataFrame:
    present = [c for c in DROP_COLUMNS if c in df.columns]
    print(f"[drop] удаляю {len(present)} колонок")
    return df.drop(columns=present)


def expand_jsonb_bools(df: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    if col not in df.columns:
        return df

    def to_dict(v):
        if v is None:
            return {}
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            import json
            try:
                return json.loads(v)
            except Exception:
                return {}
        return {}

    parsed = df[col].apply(to_dict)
    if parsed.apply(lambda d: len(d) == 0).all():
        return df.drop(columns=[col])

    expanded = pd.json_normalize(parsed).add_prefix(prefix)
    for c in expanded.columns:
        expanded[c] = expanded[c].fillna(False).astype(bool)
    print(f"[jsonb] {col}: -> {len(expanded.columns)} булевых")
    return pd.concat([df.drop(columns=[col]), expanded], axis=1)


def main():
    engine = get_engine()
    df = load_raw(engine)
    df = drop_leaks(df)
    df = expand_jsonb_bools(df, "options", prefix="opt_")
    df = expand_jsonb_bools(df, "condition_flags", prefix="cond_")

    # ── вся логика фич — в одном месте ──
    df = build_features(df)

    print(f"[save] {df.shape} -> {OUT_PATH}")
    df.to_parquet(OUT_PATH, index=False)

    # sanity check
    print("\n─── sanity check ───")
    print("shape:", df.shape)
    print("\ndtypes:", df.dtypes.value_counts().to_dict())

    key = ["price", "year", "mileage_km", "power_hp",
           "car_age", "mileage_per_year"]
    print("\nNaN по ключевым полям:")
    print(df[[c for c in key if c in df.columns]].isna().sum())

    print("\nkeyword-фичи:")
    for c in sorted(df.filter(like="kw_").columns):
        print(f"  {c}: {df[c].sum()}")


if __name__ == "__main__":
    main()