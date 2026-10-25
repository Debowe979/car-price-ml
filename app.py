"""
Streamlit UI для оценки стоимости автомобиля.
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


@st.cache_resource
def get_predictor() -> CarPricePredictor:
    return CarPricePredictor(MODELS_DIR)

@st.cache_data
def get_train_data() -> pd.DataFrame:
    return pd.read_parquet(DATA_RAW / "cars.parquet")

try:
    df_train = get_train_data()
    predictor = get_predictor()
except Exception as e:
    st.error(f"Не удалось загрузить модель или данные: {e}")
    st.stop()

BRANDS = sorted(df_train["brand"].dropna().unique().tolist())

def get_models(brand: str) -> list[str]:
    if not brand:
        return []
    return sorted(df_train[df_train["brand"] == brand]["model"].dropna().unique().tolist())

def set_prefill(car: dict, real_price: int | None) -> None:
    st.session_state["real_price"] = real_price
    
    st.session_state["w_brand"] = car.get("brand") or ""
    st.session_state["w_model"] = car.get("model") or ""
    st.session_state["w_year"] = int(car.get("year") or 2020)
    st.session_state["w_mileage"] = int(car.get("mileage_km") or 100_000)
    st.session_state["w_power"] = int(car.get("power_hp") or 150)
    st.session_state["w_engine"] = float(car.get("engine_volume_l") or 2.0)
    st.session_state["w_owners"] = int(car.get("owners_count") or 1)
    st.session_state["w_photos"] = int(car.get("photos_count") or 10)
    
    if car.get("fuel") in ["бензин", "дизель", "гибрид", "электро"]:
        st.session_state["w_fuel"] = car["fuel"]
    if car.get("transmission") in ["АКПП", "механика", "робот", "вариатор"]:
        st.session_state["w_trans"] = car["transmission"]
    if car.get("drive") in ["передний", "задний", "4WD"]:
        st.session_state["w_drive"] = car["drive"]
        
    color = car.get("color")
    valid_colors = ["черный", "белый", "серый", "серебристый", "синий", "красный", "зеленый", "коричневый", "бежевый", "желтый", "голубой", "бордовый", "оранжевый", "фиолетовый"]
    if color in valid_colors:
        st.session_state["w_color"] = color
        
    st.session_state["w_desc"] = car.get("seller_description") or ""
    
    # Принудительно обновляем чекбоксы
    for k, v in car.items():
        if k.startswith(("opt_", "cond_")):
            st.session_state[f"w_{k}"] = bool(v)


st.set_page_config(page_title="Оценка автомобиля", page_icon="🚗", layout="wide")
st.title("🚗 Оценка стоимости автомобиля")
st.caption("ML-модель оценивает цену по параметрам объявления. Введите данные или вставьте ссылку на Drom.")
st.divider()

st.subheader("1. Заполнить из Drom (или вручную ниже)")

col_url, col_btn = st.columns([4, 1])
url_input = col_url.text_input("Ссылка на объявление Drom", placeholder="https://auto.drom.ru/moscow/bmw/x5/575080427.html", label_visibility="collapsed")

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
st.subheader("2. Параметры автомобиля")

col1, col2, col3 = st.columns(3)

with col1:
    brand = st.selectbox("Марка", [""] + BRANDS, key="w_brand")
    models = get_models(brand)
    model = st.selectbox("Модель", [""] + models, key="w_model")
    year = st.number_input("Год выпуска", min_value=1990, max_value=2026, step=1, key="w_year")
    mileage = st.number_input("Пробег, км", min_value=0, max_value=1_000_000, step=1000, key="w_mileage")

with col2:
    power = st.number_input("Мощность, л.с.", min_value=30, max_value=1500, step=10, key="w_power")
    engine = st.number_input("Объём, л", min_value=0.6, max_value=8.0, step=0.1, key="w_engine")
    owners = st.number_input("Владельцев", min_value=0, max_value=10, step=1, key="w_owners")
    photos = st.number_input("Кол-во фото", min_value=0, max_value=50, step=1, key="w_photos")

with col3:
    fuel = st.selectbox("Топливо", ["бензин", "дизель", "гибрид", "электро"], key="w_fuel")
    transmission = st.selectbox("КПП", ["АКПП", "механика", "робот", "вариатор"], key="w_trans")
    drive = st.selectbox("Привод", ["передний", "задний", "4WD"], key="w_drive")
    color_options = ["черный", "белый", "серый", "серебристый", "синий", "красный", "зеленый", "коричневый", "бежевый", "желтый", "голубой", "бордовый", "оранжевый", "фиолетовый"]
    color = st.selectbox("Цвет", color_options, key="w_color")

seller_desc = st.text_area("Описание от продавца", height=100, key="w_desc")

st.subheader("3. Комплектация и состояние")

OPTIONS = [
    ("opt_has_climate", "Климат-контроль"), ("opt_has_leather", "Кожаный салон"), ("opt_has_camera", "Камера"),
    ("opt_has_navigation", "Навигация"), ("opt_has_panorama", "Панорамная крыша"), ("opt_has_sunroof", "Люк"),
    ("opt_has_parktronic", "Парктроник"), ("opt_has_cruise", "Круиз-контроль"), ("opt_has_heated_seats", "Подогрев сидений"),
    ("opt_has_heated_steering", "Подогрев руля"), ("opt_has_led_lights", "LED-фары"), ("opt_has_xenon", "Ксенон"),
    ("opt_has_keyless", "Бесключевой доступ"), ("opt_has_apple_carplay", "Apple CarPlay"), ("opt_has_multimedia", "Мультимедиа"),
    ("opt_has_alloy_wheels", "Литые диски"), ("opt_has_air_suspension", "Пневмоподвеска"), ("opt_has_massage", "Массаж сидений"),
    ("opt_has_memory_seats", "Память сидений"), ("opt_has_premium_audio", "Premium audio"), ("opt_has_roof_rails", "Рейлинги"),
    ("opt_has_third_row", "Третий ряд"), ("opt_has_tow_hitch", "Фаркоп"), ("opt_has_bluetooth", "Bluetooth"),
    ("opt_has_autopilot", "Автопилот"), ("opt_has_electric_doors", "Электропривод дверей"),
]

CONDITIONS = [
    ("cond_no_accidents", "Без ДТП"), ("cond_one_owner", "Один владелец"), ("cond_warranty", "Гарантия"),
    ("cond_dealer_serviced", "Обслуживание у дилера"), ("cond_no_investments", "Не требует вложений"), ("cond_not_damaged", "Не бит"),
    ("cond_negotiable", "Возможен торг"), ("cond_urgent_sale", "Срочная продажа"), ("cond_exchange", "Возможен обмен"),
    ("cond_fresh_import", "Свежий импорт"), ("cond_from_dealer", "От дилера"), ("cond_ev", "Электро"),
    ("cond_sold_urgently", "Продаётся срочно"),
]

# ВАЖНО: Инициализируем все ключи в session_state, чтобы Streamlit не сбрасывал их в False
for key, _ in OPTIONS + CONDITIONS:
    if f"w_{key}" not in st.session_state:
        st.session_state[f"w_{key}"] = False

st.write("**Комплектация:**")
opt_cols = st.columns(4)
options_state = {}
for i, (key, label) in enumerate(OPTIONS):
    with opt_cols[i % 4]:
        options_state[key] = st.checkbox(label, key=f"w_{key}")

st.write("**Состояние:**")
cond_cols = st.columns(4)
conditions_state = {}
for i, (key, label) in enumerate(CONDITIONS):
    with cond_cols[i % 4]:
        conditions_state[key] = st.checkbox(label, key=f"w_{key}")

st.divider()

if st.button("💰 Оценить стоимость", type="primary", use_container_width=True):
    if not brand or not model:
        st.warning("Укажи марку и модель")
    else:
        car_dict = {
            "brand": brand, "model": model, "year": year, "mileage_km": mileage,
            "power_hp": power, "engine_volume_l": engine, "fuel": fuel,
            "transmission": transmission, "drive": drive, "color": color,
            "owners_count": owners, "photos_count": photos,
            "seller_type": "Частное лицо", "seller_description": seller_desc,
            **options_state, **conditions_state,
        }

        with st.spinner("Считаем..."):
            try:
                result = predictor.predict(car_dict)
                st.session_state["result"] = {"result": result, "car": car_dict}
            except Exception as e:
                st.error(f"Ошибка предсказания: {e}")

if "result" in st.session_state:
    r = st.session_state["result"]
    result = r["result"]
    
    confidence = result.get("confidence", 0.5)
    
    # Пороги
    if confidence >= 0.75:
        level = "🟢 Высокая"
        explanation = "модель уверена в диапазоне"
    elif confidence >= 0.60:
        level = "🟡 Средняя"
        explanation = "модель видит некоторую неопределенность"
    else:
        level = "🔴 Низкая"
        explanation = "интервал слишком широкий или редкая машина"

    price = result["price"]
    low, high = result["range_80"]
    width_pct = (high - low) / price * 100

    st.divider()
    st.subheader("📊 Результат оценки")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("💰 Медианная цена", f"{price:,.0f} ₽")
    col_b.metric("📉 Нижняя граница (q10)", f"{low:,.0f} ₽")
    col_c.metric("📈 Верхняя граница (q90)", f"{high:,.0f} ₽")

    st.markdown(f"**Диапазон 80%:** {low:,.0f} — {high:,.0f} ₽  (ширина {width_pct:.1f}%)")

    if confidence >= 0.75:
        st.success(f"**Уверенность модели:** {level} ({confidence*100:.0f}%) — {explanation}")
    elif confidence >= 0.40:
        st.warning(f"**Уверенность модели:** {level} ({confidence*100:.0f}%) — {explanation}")
    else:
        st.error(f"**Уверенность модели:** {level} ({confidence*100:.0f}%) — {explanation}")

    real_price = st.session_state.get("real_price")
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
                col_y.error(f"⚠️ НИЖЕ диапазона на {(low - real_price)/price*100:.0f}%")
            else:
                col_y.error(f"⚠️ ВЫШЕ диапазона на {(real_price - high)/price*100:.0f}%")

    with st.expander("ℹ️ Что это значит?"):
        st.markdown("""
        - **Медианная цена (q50)** — точка, где 50% похожих машин дешевле, 50% дороже.
        - **Нижняя граница (q10)** — 10% похожих машин дешевле этой цены, 90% — дороже.
        - **Верхняя граница (q90)** — 90% похожих машин дешевле, 10% — дороже.
        - **Диапазон 80%** — в 80% случаев реальная цена попадает в этот интервал.
        - **Уверенность** — вероятность попадания реальной цены в диапазон. ML-модель смотрит на комбинацию фич и ширину интервала.
        """)