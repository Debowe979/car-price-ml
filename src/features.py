"""
Feature engineering для модели цены авто.

Единый источник правды: одни и те же трансформации используются в ETL
(при подготовке данных) и при инференсе (для одной карточки).
"""
import re
from typing import List

import numpy as np
import pandas as pd


# ── константы ─────────────────────────────────────────────────────────

CURRENT_YEAR = 2026
CAR_AGE_SHIFT = 1

CAT_FEATURES: List[str] = [
    "brand", "model", "generation", "fuel", "transmission",
    "drive", "color", "wheel", "seller_type", "city", "region",
]

NUM_FEATURES: List[str] = [
    "car_age", "mileage_km", "mileage_per_year", "power_hp",
    "engine_volume_l", "owners_count", "photos_count",
    "seller_age_years", "year",
]

# категориальные, которые заполняем "Unknown"
CAT_FILL_UNKNOWN = [
    "brand", "model", "generation", "fuel", "transmission",
    "drive", "color", "wheel", "seller_type", "city", "region",
]

# keyword-флаги из seller_description (7 отобранных по conditional lift)
KEYWORDS = {
    "kw_no_dtp":        r"без\s+дтп|не\s+бит|не\s+крашен|не\s+бита",
    "kw_one_owner":     r"один\s+владел|1\s+владел|один\s+хозяин",
    "kw_dealer_serv":   r"официальн[а-яё]*\s+дилер",
    "kw_max_trim":      r"максимальн[а-яё]*\s+комплектац",
    "kw_serviced":      r"полностью\s+обслужен|замена\s+масла|сервисн[а-яё]*\s+книж",
    "kw_no_complaints": r"без\s+нарекан",
    "kw_sel_poehal":    r"сел\s*.{0,3}\s*поехал",
}


# ── шаги трансформации ────────────────────────────────────────────────

def clean_color(s: pd.Series) -> pd.Series:
    """Убирает CSS-мусор из значений цвета, если проскочил."""
    return (
        s.fillna("")
         .astype(str)
         .str.replace(r"^.*}", "", regex=True)
         .str.strip()
         .str.lower()
         .replace({"": None})
    )


def add_car_age(df: pd.DataFrame) -> pd.DataFrame:
    df["car_age"] = CURRENT_YEAR - df["year"] + CAR_AGE_SHIFT
    df["mileage_per_year"] = df["mileage_km"] / df["car_age"]
    df["mileage_missing"] = df["mileage_km"].isna()
    df["is_new_car"] = (df["mileage_km"] < 500) & (df["year"] >= 2024)
    return df


def add_description_flags(df: pd.DataFrame) -> pd.DataFrame:
    if "seller_description" not in df.columns:
        for k in KEYWORDS:
            df[k] = False
        return df

    text = df["seller_description"].fillna("").str.lower()
    for name, pat in KEYWORDS.items():
        df[name] = text.str.contains(pat, regex=True, na=False)
    return df


def fill_meaningful_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Заполняет NaN там, где пропуск имеет смысл."""
    if "seller_is_new" in df.columns:
        df["seller_is_new"] = df["seller_is_new"].fillna(True).astype(bool)
    if "seller_age_years" in df.columns:
        df["seller_age_years"] = df["seller_age_years"].fillna(0)
    if "photos_count" in df.columns:
        df["photos_count"] = df["photos_count"].fillna(0)

    for c in CAT_FILL_UNKNOWN:
        if c in df.columns:
            df[c] = df[c].fillna("Unknown")

    return df


def drop_constant_bools(df: pd.DataFrame) -> pd.DataFrame:
    """Удаляет булевы колонки, где все значения одинаковые.
    
    Пропускается для single-row DataFrame — иначе все булевы удалятся
    (у одной строки nunique=1 для любой булевой колонки).
    """
    if len(df) <= 1:
        return df
    bool_cols = df.select_dtypes(include="bool").columns
    const_cols = [c for c in bool_cols if df[c].nunique() <= 1]
    if const_cols:
        df = df.drop(columns=const_cols)
    return df


# ── публичная функция ─────────────────────────────────────────────────

def build_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Полный пайплайн фич-инжиниринга. Применяется и в ETL, и при инференсе.

    Вход:  сырой DataFrame (или одна строка) с колонками из БД.
    Выход: DataFrame с добавленными фичами.
    """
    df = df.copy()

    # 1. чистка категориальных
    if "color" in df.columns:
        before = df["color"].nunique(dropna=False)
        df["color"] = clean_color(df["color"])
        if verbose:
            print(f"[features] color: {before} → {df['color'].nunique(dropna=False)}")

    # 2. keyword-флаги
    if verbose:
        n_kw = len([c for c in df.columns if c.startswith("kw_")]) or len(KEYWORDS)
        print(f"[features] keyword-флаги: {n_kw}")
    df = add_description_flags(df)

    # 3. заполнение осмысленных NaN
    df = fill_meaningful_nulls(df)

    # 4. числовые фичи
    df = add_car_age(df)

    # 5. удаление константных булевых
    df = drop_constant_bools(df)

    return df


def get_feature_lists(df: pd.DataFrame):
    """
    Возвращает списки фич по типам с учётом того, что реально есть в df.
    Используется при обучении и инференсе.
    """
    bool_features = [
        c for c in df.select_dtypes(include="bool").columns
        if c != "mileage_missing"  # оставим — она тоже bool
    ]
    num = [c for c in NUM_FEATURES if c in df.columns]
    cat = [c for c in CAT_FEATURES if c in df.columns]

    # все фичи
    features = num + bool_features + cat
    features = [f for f in features if f in df.columns]
    # убрать дубликаты
    features = list(dict.fromkeys(features))

    return features, num, bool_features, cat