"""Обёртка над парсером Drom: URL → dict для модели."""
from .detail_page import parse_detail_page
from .http_client import HttpClient, CaptchaDetected


# Список опций, как в api.py CarInput (26 без ventilated_seats)
ALL_OPTIONS = [
    "has_xenon", "has_camera", "has_cruise", "has_climate", "has_keyless",
    "has_leather", "has_massage", "has_sunroof", "has_panorama", "has_autopilot",
    "has_bluetooth", "has_third_row", "has_tow_hitch", "has_led_lights",
    "has_multimedia", "has_navigation", "has_parktronic", "has_roof_rails",
    "has_alloy_wheels", "has_heated_seats", "has_memory_seats",
    "has_apple_carplay", "has_premium_audio", "has_air_suspension",
    "has_electric_doors", "has_heated_steering",
]

# Список condition_flags, как в api.py CarInput (13)
ALL_CONDITIONS = [
    "ev", "exchange", "warranty", "one_owner", "negotiable",
    "from_dealer", "not_damaged", "urgent_sale", "fresh_import",
    "no_accidents", "sold_urgently", "no_investments", "dealer_serviced",
]

# Поля из парсера, которые НЕ идут в модель
# (утечки + служебные, которые развернём в opt_*/cond_*)
DROP_BEFORE_MODEL = [
    # цена продавца — таргет, в модель нельзя
    "price",
    # служебные
    "url", "id", "engine_raw", "vin_masked",
    # утечки
    "market_price", "deal_type", "views", "published_date",
    "json_ld", "bull_data", "price_discount",
    # не нужно модели
    "photos",  # остаётся photos_count
    "modification_id",
    # развернём в opt_* и cond_*
    "options", "condition_flags",
]


def _flatten_booleans(raw: dict | None, keys: list[str], prefix: str) -> dict:
    """{"has_leather": True} → {"opt_has_leather": True, "opt_has_camera": False, ...}"""
    raw = raw or {}
    return {f"{prefix}{k}": bool(raw.get(k, False)) for k in keys}


def normalize_for_model(parsed: dict) -> dict:
    """Превращает ответ parse_detail_page в формат CarInput."""
    car = {k: v for k, v in parsed.items() if k not in DROP_BEFORE_MODEL}

    car.update(_flatten_booleans(parsed.get("options"), ALL_OPTIONS, "opt_"))
    car.update(_flatten_booleans(parsed.get("condition_flags"), ALL_CONDITIONS, "cond_"))

    return car


def parse_drom_url(url: str, client: HttpClient | None = None) -> dict:
    if not url or "drom.ru" not in url:
        raise ValueError("URL должен быть с auto.drom.ru")

    if client is None:
        client = HttpClient()

    html = client.get_html(url)
    if not html:
        raise ValueError(f"Не удалось скачать {url} (404 или пустой ответ)")

    parsed = parse_detail_page(html, url)

    return {
        "car": normalize_for_model(parsed),
        "real_price": parsed.get("price"),
        "raw": parsed,
    }