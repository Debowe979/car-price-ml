import requests
import time

API = "http://localhost:8000"

# 1. health
r = requests.get(f"{API}/health")
print(f"Health: {r.json()}")

# 2. тест 3 машин
cars = [
    {
        "brand": "Лада", "model": "Гранта", "year": 2018,
        "mileage_km": 120000, "power_hp": 106, "engine_volume_l": 1.6,
        "fuel": "бензин", "transmission": "механика", "drive": "передний",
        "color": "белый", "owners_count": 2, "seller_type": "Частное лицо",
    },
    {
        "brand": "Toyota", "model": "Camry", "year": 2020,
        "mileage_km": 60000, "power_hp": 200, "engine_volume_l": 2.5,
        "fuel": "бензин", "transmission": "АКПП", "drive": "передний",
        "color": "черный", "owners_count": 1, "seller_type": "Частное лицо",
        "seller_description": "Один владелец, не бит",
    },
    {
        "brand": "BMW", "model": "X5", "year": 2020,
        "mileage_km": 80000, "power_hp": 340, "engine_volume_l": 3.0,
        "fuel": "бензин", "transmission": "АКПП", "drive": "4WD",
        "color": "черный", "owners_count": 2, "seller_type": "Частное лицо",
        "opt_has_panorama": True, "opt_has_leather": True,
        "opt_has_navigation": True, "opt_has_camera": True,
        "cond_no_accidents": True,
    },
]

for car in cars:
    t0 = time.time()
    r = requests.post(f"{API}/predict", json=car, timeout=30)
    elapsed = (time.time() - t0) * 1000
    data = r.json()
    print(f"\n{car['brand']} {car['model']} {car['year']}:")
    print(f"  Цена:     {data['price']:>12,.0f} ₽")
    print(f"  Диапазон: [{data['range_80_low']:>10,.0f} — {data['range_80_high']:>10,.0f}] ₽")
    print(f"  Ширина:   {data['range_width_pct']:.1f}%")
    print(f"  Время:    {elapsed:.0f} ms")