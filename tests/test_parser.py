"""Тест парсера Drom + сравнение с ML-моделью."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.drom_parser import parse_drom_url


def main():
    # Вставь свой реальный URL
    url = "https://auto.drom.ru/moscow/bmw/x5/575080427.html"

    print(f"Парсим: {url}")
    try:
        result = parse_drom_url(url)
    except Exception as e:
        print(f"Ошибка парсинга: {e}")
        return

    car = result["car"]
    print(f"\nУспешно распарсено полей: {len(car)}")
    print(f"Реальная цена продавца: {result['real_price']:,} ₽")

    print("\nОсновные поля:")
    for key in ["brand", "model", "year", "mileage_km", "power_hp",
                "engine_volume_l", "fuel", "transmission", "drive",
                "color", "owners_count", "seller_type", "photos_count",
                "seller_age_years", "seller_is_new"]:
        print(f"  {key:20s}: {car.get(key)}")

    opts = [k for k in car if k.startswith("opt_") and car[k]]
    print(f"\nНайдено опций: {len(opts)}")
    for k in opts:
        print(f"  {k}")

    conds = [k for k in car if k.startswith("cond_") and car[k]]
    print(f"\nНайдено флагов состояния: {len(conds)}")
    for k in conds:
        print(f"  {k}")

    # --- Сравнение с моделью ---
    print("\n" + "=" * 60)
    print("Сравнение с ML-моделью")
    print("=" * 60)

    from src.db import MODELS_DIR
    from src.predict import CarPricePredictor

    predictor = CarPricePredictor(MODELS_DIR)
    pred = predictor.predict(car)

    print(f"Цена продавца:    {result['real_price']:>14,.0f} ₽")
    print(f"ML-оценка (q50):  {pred['price']:>14,.0f} ₽")
    print(f"Диапазон 80%:     [{pred['range_80'][0]:>12,.0f} — {pred['range_80'][1]:>12,.0f}] ₽")

    if result["real_price"]:
        diff = (result["real_price"] - pred["price"]) / pred["price"] * 100
        covered = pred["range_80"][0] <= result["real_price"] <= pred["range_80"][1]
        print(f"Отклонение цены:  {diff:>+13.1f}%")
        print(f"В интервале:      {'✅ да' if covered else '❌ нет'}")


if __name__ == "__main__":
    main()