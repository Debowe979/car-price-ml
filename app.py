"""
Streamlit UI для оценки стоимости автомобиля.

Запуск:
    streamlit run app.py
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.db import DATA_RAW, MODELS_DIR
from src.drom_parser import parse_drom_url
from src.predict import CarPricePredictor


# ── кэшированные загрузки ────────────────────────────────────────────

@st.cache_resource
def get_predictor() -> CarPricePredictor:
    return CarPricePredictor(MODELS_DIR)


@st.cache_data
def get_train_data() -> pd.DataFrame:
    return pd.read_parquet(DATA_RAW / "cars.parquet")


# ── уверенность (эвристика на 3 факторах) ────────────────────────────

def estimate_confidence(result: dict, car: dict, df_train: pd.DataFrame) -> tuple[float, str, str]:
    """
    Возвращает (score 0..1, уровень, объяснение).
    v1 — эвристика; v2 — ML-классификатор (см. план).
    """
    price = result["price"]
    low, high = result["range_80"]
    width_pct = (high - low) / price * 100 if price else 0

    # 1. Ширина интервала: узкий → уверенно
    if width_pct < 10:
        width_score = 1.0
    elif width_pct < 20:
        width_score = 0.85
    elif width_pct < 35:
        width_score = 0.70
    elif width_pct < 50:
        width_score = 0.55
    else:
        width_score = 0.40

    # 2. Сколько похожих машин в train
    brand = car.get("brand")
    model = car.get("model")
    brand_mask = df_train["brand"] == brand if brand else pd.Series(False, index=df_train.index)
    if model:
        model_mask = brand_mask & (df_train["model"] == model)
    else:
        model_mask = brand_mask
    model_count = int(model_mask.sum())

    if model_count >= 200:
        data_score = 1.0
    elif model_count >= 50:
        data_score = 0.85
    elif model_count >= 20:
        data_score = 0.70
    elif model_count >= 5:
        data_score = 0.50
    else:
        data_score = 0.30

    # 3. Свежие годы — мало в train
    year = car.get("year", 2020)
    if year <= 2020:
        age_score = 1.0
    elif year == 2021:
        age_score = 0.90
    elif year == 2022:
        age_score = 0.80
    elif year == 2023:
        age_score = 0.70
    elif year >= 2024:
        age_score = 0.55
    else:
        age_score = 0.5

    # Комбинация
    score = 0.5 * width_score + 0.3 * data_score + 0.2 * age_score

    if score >= 0.80:
        level = "🟢 Высокая"
    elif score >= 0.60:
        level = "🟡 Средняя"
    else:
        level = "🔴 Низкая"

    reasons = []
    if width_score < 0.7:
        reasons.append(f"широкий интервал ({width_pct:.0f}%)")
    if data_score < 0.7:
        reasons.append(f"мало похожих машин в данных ({model_count})")
    if age_score < 0.7:
        reasons.append(f"свежий год ({year}) — модель реже их видела")

    explanation = "; ".join(reasons) if reasons else "все факторы в норме"
    return score, level, explanation


# ── загрузка данных ──────────────────────────────────────────────────

try:
    df_train = get_train_data()
    predictor = get_predictor()
except Exception as e:
    st.error(f"Не удалось загрузить модель или данные: {e}")
    st.stop()

BRANDS = sorted(df_train["brand"].dropna().unique().tolist())


# ── helpers для виджетов ─────────────────────────────────────────────

def get_models(brand: str) -> list[str]:
    if not brand:
        return []
    return sorted(df_train[df_train["brand"] == brand]["model"].dropna().unique().tolist())


def set_prefill(car: dict, real_price: int | None) -> None:
    """Запоминает данные из парсера в session_state перед rerun."""
    st.session_state["prefill"] = {
        "brand": car.get("brand"),
        "model": car.get("model"),
        "year": car.get("year"),
        "mileage_km": car.get("mileage_km"),
        "power_hp": car.get("power_hp"),
        "engine_volume_l": car.get("engine_volume_l"),
        "fuel": car.get("fuel"),
        "transmission": car.get("transmission"),
        "drive": car.get("drive"),
        "color": car.get("color"),
        "owners_count": car.get("owners_count"),
        "seller_type": car.get("seller_type"),
        "seller_description": car.get("seller_description") or "",
        "real_price": real_price,
    }
    # булевы опции
    for k, v in car.items():
        if k.startswith(("opt_", "cond_")):
            st.session_state["prefill"][k] = bool(v)


def prefill_get(key: str, default=None):
    """Берёт значение из prefill, если оно есть."""
    return st.session_state.get("prefill", {}).get(key, default)


# ── HEADER ────────────────────────────────────────────────────────────

st.set_page_config(page_title="Оценка автомобиля", page_icon="🚗", layout="wide")
st.title("🚗 Оценка стоимости автомобиля")
st.caption("ML-модель оценивает цену по параметрам объявления. Введите данные или вставьте ссылку на Drom.")

st.divider()


# ── URL PARSER ────────────────────────────────────────────────────────

st.subheader("1. Заполнить из Drom (или вручную ниже)")

col_url, col_btn = st.columns([4, 1])
url_input = col_url.text_input(
    "Ссылка на объявление Drom",
    placeholder="https://auto.drom.ru/moscow/bmw/x5/575080427.html",
    label_visibility="collapsed",
)
if col_btn.button("🔗 Заполнить", use_container_width=True):
    if not url_input or "drom.ru" not in url_input:
        st.error("Вставь корректную ссылку на auto.drom.ru")
    else:
        with st.spinner("Парсим объявление..."):
            try:
                parsed = parse_drom_url(url_input)
                set_prefill(parsed["car"], parsed["real_price"])
                st.success(f"Готово: {parsed['car'].get('brand')} {parsed['car'].get('model')} {parsed['car'].get('year')}")
                st.rerun()
            except Exception as e:
                st.error(f"Не удалось распарсить: {e}")

st.divider()


# ── MAIN FORM ────────────────────────────────────────────────────────

st.subheader("2. Параметры автомобиля")

col1, col2, col3 = st.columns(3)

with col1:
    default_brand = prefill_get("brand")
    brand_index = BRANDS.index(default_brand) if default_brand in BRANDS else 0
    brand = st.selectbox("Марка", [""] + BRANDS, index=brand_index, key="w_brand")

    models = get_models(brand)
    default_model = prefill_get("model")
    model_index = models.index(default_model) if default_model in models else 0
    if models:
        model = st.selectbox("Модель", [""] + models, index=model_index, key="w_model")
    else:
        model = ""

    year = st.number_input(
        "Год выпуска", min_value=1990, max_value=2026,
        value=prefill_get("year") or 2020, step=1, key="w_year",
    )

    mileage = st.number_input(
        "Пробег, км", min_value=0, max_value=1_000_000,
        value=int(prefill_get("mileage_km") or 100_000), step=1000, key="w_mileage",
    )

with col2:
    power = st.number_input(
        "Мощность, л.с.", min_value=30, max_value=1500,
        value=int(prefill_get("power_hp") or 150), step=10, key="w_power",
    )

    engine = st.number_input(
        "Объём, л", min_value=0.6, max_value=8.0,
        value=float(prefill_get("engine_volume_l") or 2.0), step=0.1, key="w_engine",
    )

    owners = st.number_input(
        "Владельцев", min_value=0, max_value=10,
        value=int(prefill_get("owners_count") or 1), step=1, key="w_owners",
    )

    photos = st.number_input(
        "Кол-во фото", min_value=0, max_value=50, value=10, step=1, key="w_photos",
    )

with col3:
    fuel_options = ["бензин", "дизель", "гибрид", "электро"]
    fuel = st.selectbox("Топливо", fuel_options, index=0, key="w_fuel")

    trans_options = ["АКПП", "механика", "робот", "вариатор"]
    transmission = st.selectbox("КПП", trans_options, index=0, key="w_trans")

    drive_options = ["передний", "задний", "4WD"]
    drive = st.selectbox("Привод", drive_options, index=0, key="w_drive")

    color_options = ["черный", "белый", "серый", "серебристый", "синий",
                     "красный", "зеленый", "коричневый", "бежевый", "желтый",
                     "голубой", "бордовый", "оранжевый", "фиолетовый"]
    default_color = prefill_get("color")
    color_index = color_options.index(default_color) if default_color in color_options else 0
    color = st.selectbox("Цвет", color_options, index=color_index, key="w_color")

# Описание
seller_desc = st.text_area(
    "Описание от продавца",
    value=prefill_get("seller_description") or "",
    height=100,
    placeholder="Один владелец, не бит, полностью обслужен...",
    key="w_desc",
)


# ── OPTIONS ──────────────────────────────────────────────────────────

st.subheader("3. Комплектация и состояние")

OPTIONS = [
    ("opt_has_climate", "Климат-контроль"),
    ("opt_has_leather", "Кожаный салон"),
    ("opt_has_camera", "Камера"),
    ("opt_has_navigation", "Навигация"),
    ("opt_has_panorama", "Панорамная крыша"),
    ("opt_has_sunroof", "Люк"),
    ("opt_has_parktronic", "Парктроник"),
    ("opt_has_cruise", "Круиз-контроль"),
    ("opt_has_heated_seats", "Подогрев сидений"),
    ("opt_has_heated_steering", "Подогрев руля"),
    ("opt_has_led_lights", "LED-фары"),
    ("opt_has_xenon", "Ксенон"),
    ("opt_has_keyless", "Бесключевой доступ"),
    ("opt_has_apple_carplay", "Apple CarPlay"),
    ("opt_has_multimedia", "Мультимедиа"),
    ("opt_has_alloy_wheels", "Литые диски"),
    ("opt_has_air_suspension", "Пневмоподвеска"),
    ("opt_has_massage", "Массаж сидений"),
    ("opt_has_memory_seats", "Память сидений"),
    ("opt_has_premium_audio", "Premium audio"),
    ("opt_has_roof_rails", "Рейлинги"),
    ("opt_has_third_row", "Третий ряд"),
    ("opt_has_tow_hitch", "Фаркоп"),
    ("opt_has_bluetooth", "Bluetooth"),
    ("opt_has_autopilot", "Автопилот"),
    ("opt_has_electric_doors", "Электропривод дверей"),
]

CONDITIONS = [
    ("cond_no_accidents", "Без ДТП"),
    ("cond_one_owner", "Один владелец"),
    ("cond_warranty", "Гарантия"),
    ("cond_dealer_serviced", "Обслуживание у дилера"),
    ("cond_no_investments", "Не требует вложений"),
    ("cond_not_damaged", "Не бит"),
    ("cond_negotiable", "Возможен торг"),
    ("cond_urgent_sale", "Срочная продажа"),
    ("cond_exchange", "Возможен обмен"),
    ("cond_fresh_import", "Свежий импорт"),
    ("cond_from_dealer", "От дилера"),
    ("cond_ev", "Электро"),
    ("cond_sold_urgently", "Продаётся срочно"),
]

# Опции — колонками
st.write("**Комплектация:**")
opt_cols = st.columns(4)
options_state = {}
for i, (key, label) in enumerate(OPTIONS):
    with opt_cols[i % 4]:
        options_state[key] = st.checkbox(label, value=prefill_get(key, False), key=f"w_{key}")

st.write("**Состояние:**")
cond_cols = st.columns(4)
conditions_state = {}
for i, (key, label) in enumerate(CONDITIONS):
    with cond_cols[i % 4]:
        conditions_state[key] = st.checkbox(label, value=prefill_get(key, False), key=f"w_{key}")


# ── PREDICT ──────────────────────────────────────────────────────────

st.divider()

real_price = prefill_get("real_price")

if st.button("💰 Оценить стоимость", type="primary", use_container_width=True):
    if not brand or not model:
        st.warning("Укажи марку и модель")
    else:
        car_dict = {
            "brand": brand,
            "model": model,
            "year": year,
            "mileage_km": mileage,
            "power_hp": power,
            "engine_volume_l": engine,
            "fuel": fuel,
            "transmission": transmission,
            "drive": drive,
            "color": color,
            "owners_count": owners,
            "photos_count": photos,
            "seller_type": "Частное лицо",
            "seller_description": seller_desc,
            **options_state,
            **conditions_state,
        }

        with st.spinner("Считаем..."):
            try:
                result = predictor.predict(car_dict)
                confidence, level, explanation = estimate_confidence(result, car_dict, df_train)
                st.session_state["result"] = {
                    "result": result,
                    "confidence": confidence,
                    "level": level,
                    "explanation": explanation,
                    "car": car_dict,
                }
            except Exception as e:
                st.error(f"Ошибка предсказания: {e}")


# ── RESULT ───────────────────────────────────────────────────────────

if "result" in st.session_state:
    r = st.session_state["result"]
    result = r["result"]
    confidence = r["confidence"]
    level = r["level"]
    explanation = r["explanation"]

    price = result["price"]
    low, high = result["range_80"]
    width_pct = (high - low) / price * 100

    st.divider()
    st.subheader("📊 Результат оценки")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("💰 Медианная цена", f"{price:,.0f} ₽")
    col_b.metric("📉 Оптимистично (q10)", f"{low:,.0f} ₽")
    col_c.metric("📈 Пессимистично (q90)", f"{high:,.0f} ₽")

    st.markdown(f"**Диапазон 80%:** {low:,.0f} — {high:,.0f} ₽  (ширина {width_pct:.1f}%)")

    # Уверенность
    if confidence >= 0.80:
        st.success(f"**Уверенность модели:** {level} ({confidence*100:.0f}%) — {explanation}")
    elif confidence >= 0.60:
        st.warning(f"**Уверенность модели:** {level} ({confidence*100:.0f}%) — {explanation}")
    else:
        st.error(f"**Уверенность модели:** {level} ({confidence*100:.0f}%) — {explanation}")

    # Сравнение с реальной ценой (если парсили)
    if real_price:
        st.markdown("---")
        st.markdown("**Сравнение с ценой продавца:**")
        diff_pct = (real_price - price) / price * 100
        covered = low <= real_price <= high

        col_x, col_y = st.columns(2)
        col_x.metric("Продавец просит", f"{real_price:,.0f} ₽", f"{diff_pct:+.1f}% от ML-оценки")

        if covered:
            col_y.success("✅ В диапазоне модели")
        else:
            if real_price < low:
                col_y.error(f"⚠️ НИЖЕ диапазона на {(low - real_price)/price*100:.0f}% — возможно, есть скрытые дефекты")
            else:
                col_y.error(f"⚠️ ВЫШЕ диапазона на {(real_price - high)/price*100:.0f}% — продавец завышает")

    with st.expander("ℹ️ Что это значит?"):
        st.markdown("""
        - **Медианная цена (q50)** — точка, где 50% похожих машин дешевле, 50% дороже.
        - **Оптимистично (q10)** — 10% похожих машин дешевле, 90% дороже. Быстрая продажа.
        - **Пессимистично (q90)** — 90% дешевле, 10% дороже. Максимальная цена при долгом ожидании.
        - **Диапазон 80%** — в 80% случаев реальная цена попадает в этот интервал.
        - **Уверенность** — оценка того, насколько можно доверять предсказанию. Зависит от ширины интервала, количества похожих машин в обучающих данных и свежести года.
        """)