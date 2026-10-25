import json
import re
from datetime import datetime

from selectolax.lexbor import LexborHTMLParser as HTMLParser


def _clean(node):
    if not node:
        return None
    text = node.text(strip=True)
    if not text:
        return None
    return re.sub(r"\s+", " ", text).strip()


def _to_int(s):
    if s is None:
        return None
    if isinstance(s, int):
        return s
    digits = re.sub(r"\D", "", str(s))
    return int(digits) if digits else None


def _to_float(s):
    if s is None:
        return None
    m = re.search(r"\d+[.,]?\d*", str(s).replace(",", "."))
    return float(m.group()) if m else None

def _extract_seller_age_years(tree: HTMLParser) -> int | None:
    """
    Извлекает 'N лет на Дроме' из блока продавца.
    
    Примеры HTML:
        'Частное лицо1 год на Дроме'
        'Частное лицо Более 10 лет на Дроме'
        'Дилер 5 лет на Дроме'
    
    Возвращает:
        10   — для 'Более 10 лет'
        1    — для '1 год'
        5    — для '5 лет'
        None — если блок не найден или фраза не распознана
    """
    node = tree.css_first('[data-ftid="bulletin-card_private"]')
    if not node:
        # Попробуем альтернативный селектор — вдруг это дилер
        node = tree.css_first('[data-ftid="bulletin-card_dealer"]')
    if not node:
        return None

    text = node.text(strip=True).lower()

    # 'более 10 лет' → 10
    m = re.search(r"более\s+(\d+)\s*(?:год|лет|года)", text)
    if m:
        return int(m.group(1))

    # '1 год', '5 лет', '3 года'
    m = re.search(r"(\d+)\s*(?:год|лет|года)", text)
    if m:
        return int(m.group(1))

    # '9 месяцев' → 0 (меньше года)
    m = re.search(r"(\d+)\s*месяц", text)
    if m:
        return 0

    # 'меньше года' / 'менее года' → 0
    if "меньше года" in text or "менее года" in text:
        return 0

    return None

#New part

TRANSMISSION_MAP = {
    "акпп": "АКПП",
    "автомат": "АКПП",
    "автоматическая": "АКПП",
    "механика": "механика",
    "мкпп": "механика",
    "робот": "робот",
    "роботизированная": "робот",
    "вариатор": "вариатор",
    "cvt": "вариатор",
}

def _normalize_transmission(s: str | None) -> str | None:
    if not s:
        return None
    low = s.strip().lower()
    return TRANSMISSION_MAP.get(low, s.strip())

DRIVE_MAP = {
    "передний": "передний",
    "задний": "задний",
    "4wd": "4WD",
    "полный": "4WD",
    "awd": "4WD",
}

COLOR_MAP = {
    "черный": "черный", "чёрный": "черный", "черн": "черный",
    "белый": "белый", "бел": "белый",
    "серебристый": "серебристый", "серебро": "серебристый", "серебр": "серебристый",
    "серый": "серый", "сер": "серый",
    "красный": "красный", "красн": "красный",
    "синий": "синий", "син": "синий",
    "голубой": "голубой", "голуб": "голубой",
    "зеленый": "зеленый", "зелён": "зеленый", "зелен": "зеленый",
    "бежевый": "бежевый", "беж": "бежевый",
    "коричневый": "коричневый", "коричн": "коричневый",
    "золотой": "золотой", "золот": "золотой",
    "оранжевый": "оранжевый", "оранж": "оранжевый",
    "желтый": "желтый", "жёлтый": "желтый", "желт": "желтый",
    "фиолетовый": "фиолетовый", "фиолет": "фиолетовый",
    "бордовый": "бордовый", "бордо": "бордовый",
    "вишневый": "вишневый", "вишн": "вишневый",
    "розовый": "розовый", "розов": "розовый",
}

def _normalize_color(s: str | None) -> str | None:
    if not s:
        return None
    low = s.strip().lower()
    # Сначала точное совпадение
    if low in COLOR_MAP:
        return COLOR_MAP[low]
    # Потом поиск подстроки — для «серебристый металлик», «темно-серый»
    for key, val in COLOR_MAP.items():
        if key in low:
            return val
    return s.strip()

def _normalize_drive(s: str | None) -> str | None:
    if not s:
        return None
    low = s.strip().lower()
    return DRIVE_MAP.get(low, s.strip())

def _parse_mileage(s):
    """
    '139 000 км'  → 139000
    '139 тыс. км' → 139000
    '500 км'      → 500
    """
    if not s:
        return None
    low = s.lower()
    m = re.search(r"(\d[\d\s]{0,7})", low)
    if not m:
        return None
    digits = re.sub(r"\D", "", m.group(1))
    val = int(digits) if digits else None
    if val is not None and "тыс" in low:
        val *= 1000
    return val

#FUEL_KEYWORDS = {
#    "бензин": "бензин",
#    "дизель": "дизель",
#    "электро": "электро",
#    "гибрид": "гибрид",
#    "газ": "газ",
#    "бензин-газ": "газ",
#    "дизель-газ": "газ",
#}

#FUEL_ORDER = ["бензин", "дизель", "электро", "гибрид", "газ"]


FUEL_PRIORITY = ["гибрид", "дизель", "электро", "газ", "бензин"]

def _extract_fuel(engine_raw: str | None) -> str | None:
    """
    Приоритетный парсинг: если в строке есть гибрид — возвращаем гибрид,
    даже если он указан после бензина.
    """
    if not engine_raw:
        return None
    low = engine_raw.lower()
    for fuel in FUEL_PRIORITY:
        if fuel in low:
            return fuel
    return None


def _extract_engine_volume(s: str | None) -> float | None:
    """
    'бензин, 3.0 л' → 3.0
    '1.5 л'         → 1.5
    """
    if not s:
        return None
    m = re.search(r"(\d+[.,]?\d*)\s*л", s)
    return float(m.group(1).replace(",", ".")) if m else None

def _extract_power_hp(s):
    """
    Из строки вида '258 л.с. , налог' или '249 л.с.' достаёт 258/249.
    Если числа нет — None.
    """
    if not s:
        return None
    m = re.search(r"(\d[\d\s]{0,5})\s*л\.?\s*с\.?", s)
    if not m:
        return None
    digits = re.sub(r"\D", "", m.group(1))
    return int(digits) if digits else None


def _extract_id_from_url(url: str):
    if not url:
        return None
    m = re.search(r"/(\d+)\.html", url)
    return int(m.group(1)) if m else None


def _extract_json_ld(tree: HTMLParser):
    for script in tree.css('script[type="application/ld+json"]'):
        try:
            data = json.loads(script.text())
        except Exception:
            continue
        if isinstance(data, dict) and data.get("@type") == "Car":
            return data
    return None


def _extract_bull_data(tree: HTMLParser):
    node = tree.css_first('script[data-drom-module="bull-page"]')
    if not node:
        return None
    try:
        return json.loads(node.text())
    except Exception:
        return None


def _extract_price_discount(tree: HTMLParser) -> int | None:
    """
    Ищет информацию о снижении цены.
    Пример: 'Цена снижена на 150 000 ₽'
    """
    # Ищем текст с "снижен" в блоке цены
    price_node = tree.css_first('[data-ftid="bulletin-price"]')
    if price_node:
        # Проверяем соседние элементы
        parent = price_node.parent
        if parent:
            text = parent.text(strip=True)
            m = re.search(r"снижен[ао]?\s*на\s*([\d\s]+)", text, re.IGNORECASE)
            if m:
                digits = re.sub(r"\D", "", m.group(1))
                return int(digits) if digits else None

    # Альтернативный поиск по всему документу
    for node in tree.css("div, span, p"):
        text = node.text(strip=True)
        if "снижен" in text.lower() and "цен" in text.lower():
            m = re.search(r"([\d\s]{5,})", text)
            if m:
                digits = re.sub(r"\D", "", m.group(1))
                if digits:
                    return int(digits)

    return None


def parse_detail_page(html: str, url: str) -> dict:
    tree = HTMLParser(html)
    data = {"url": url, "id": _extract_id_from_url(url)}

    # --- Цена ---
    data["price"] = _to_int(_clean(tree.css_first('[data-ftid="bulletin-price"]')))

    # --- Таблица характеристик ---
    specs = {}
    table = tree.css_first('table[data-ftid="bulletin-specifications"]')
    if table:
        for tr in table.css("tr[data-ftid]"):
            th = tr.css_first("th")
            td = tr.css_first("td")
            if th and td:
                key = _clean(th)
                val = _clean(td)
                if key:
                    specs[key] = val

    data["year"] = _to_int(specs.get("Год выпуска"))
    data["mileage_km"] = _parse_mileage(specs.get("Пробег"))
    data["owners_count"] = _to_int(specs.get("Владельцы"))
    data["color"] = _normalize_color(specs.get("Цвет"))
    data["wheel"] = specs.get("Руль")
    #data["drive"] = specs.get("Привод")
    data["drive"] = _normalize_drive(specs.get("Привод"))
    data["transmission"] = _normalize_transmission(specs.get("Коробка передач"))
    #data["transmission"] = specs.get("Коробка передач")
    data["generation"] = specs.get("Поколение")
    data["power_hp"] = _extract_power_hp(specs.get("Мощность"))

    # --- Двигатель ---
    engine_raw = specs.get("Двигатель")
    data["engine_raw"] = engine_raw
    data["fuel"] = _extract_fuel(engine_raw)
    data["engine_volume_l"] = _extract_engine_volume(engine_raw)

    # --- Город ---
    data["city"] = _clean(tree.css_first('[data-ftid="city"] [data-ftid="value"]'))

    # --- Продавец ---
    data["seller_type"] = _clean(tree.css_first('[data-ftid="bulletin-card_title"]'))

    # --- Описание продавца ---
    desc_node = tree.css_first('[data-ftid="info-full"] [data-ftid="value"]')
    data["seller_description"] = (
        desc_node.text(separator="\n", strip=True) if desc_node else None
    )

    # --- VIN (маскированный) ---
    vin_match = re.search(r"([A-Z0-9]{3}\*{6,})", html)
    data["vin_masked"] = vin_match.group(1) if vin_match else None

    # --- JSON-LD: brand / model / сырой JSON ---
    jld = _extract_json_ld(tree)
    if jld:
        brand = jld.get("brand")
        if isinstance(brand, dict):
            data["brand"] = brand.get("name")
        elif isinstance(brand, str):
            data["brand"] = brand
        else:
            data["brand"] = None

        data["model"] = jld.get("model") or jld.get("name")
        data["json_ld"] = jld
    else:
        data["brand"] = None
        data["model"] = None
        data["json_ld"] = None

    # --- bullData: фото + market price + views + published_date ---
    bull = _extract_bull_data(tree)
    data["bull_data"] = None
    data["photos"] = []
    data["photos_count"] = 0
    data["market_price"] = None
    data["deal_type"] = None
    data["views"] = None
    data["published_date"] = None

    if bull:
        data["bull_data"] = {
            "constants": bull.get("constants"),
            "priceWidgetData": bull.get("priceWidgetData"),
            "bullCounters": bull.get("bullCounters"),
        }

        images = (
            bull.get("gallery", {})
                .get("photos", {})
                .get("images", [])
        )
        photos = [
            img["image"]["src"]
            for img in images
            if img.get("image", {}).get("src")
        ]

        if not photos:
            for a in tree.css('a[data-ga-stats-name="gallery-thumbnail"]'):
                href = a.attributes.get("href")
                if href:
                    photos.append(href)

        data["photos"] = photos
        data["photos_count"] = len(photos)

        gd = (
            bull.get("priceWidgetData", {})
                .get("goodDealConfig", {})
        )
        data["market_price"] = gd.get("marketPrice")
        data["deal_type"] = gd.get("dealType")

        counters = bull.get("bullCounters", {})
        data["views"] = counters.get("totalViews")
        data["published_date"] = counters.get("bullDate")

        # --- Дополнительные фичи ---
    # 1. Опции комплектации
    data["options"] = _extract_options(data.get("seller_description"))

    # 2. Снижение цены
    data["price_discount"] = _extract_price_discount(tree)

    # 4. Флаги состояния из описания
    data["condition_flags"] = _extract_condition_flags(data.get("seller_description"))

    # 5. Расширенные счётчики из bullData
    data["modification_id"] = None

    # --- Возраст продавца на Drom ---
    data["seller_age_years"] = _extract_seller_age_years(tree)

    age = data.get("seller_age_years")
    data["seller_is_new"] = (age == 0) if age is not None else None

    return data

def _extract_options(description: str | None) -> dict:
    """
    Извлекает булевы фичи из текста описания.
    Работает только по seller_description, не по всему HTML.
    """
    if not description:
        return {}

    low = description.lower()
    return {
        "has_leather": any(k in low for k in ["кожа", "кожан", "наппа"]),
        "has_sunroof": any(k in low for k in ["люк", "панорам"]),
        "has_camera": any(k in low for k in ["камера", "видеокамер", "360"]),
        "has_parktronic": any(k in low for k in ["парктроник", "парковочн"]),
        "has_cruise": any(k in low for k in ["круиз"]),
        "has_navigation": any(k in low for k in ["навигаци", "navi"]),
        "has_heated_seats": any(k in low for k in ["подогрев сиден", "обогрев сиден","подогрев передн", "подогрев задн","обогрев передн", "обогрев задн","подогревы передн", "подогревы задн"]),
        "has_heated_steering": any(k in low for k in ["подогрев рул", "обогрев рул"]),
        "has_bluetooth": any(k in low for k in ["bluetooth", "блютуз"]),
        "has_climate": any(k in low for k in ["климат-контроль", "климат контроль", "двухзонн", "трехзонн", "четырехзонн"]),
        "has_air_suspension": any(k in low for k in ["пневмоподвеск", "пневмо подвеск", "пневматическ"]),
        "has_panorama": any(k in low for k in ["панорам"]),
        "has_electric_doors": any(k in low for k in ["электропривод двер", "электропривод багажник"]),
        "has_autopilot": any(k in low for k in ["автопилот", "autopilot", "ассистент вожден"]),
        "has_alloy_wheels": any(k in low for k in ["литые диски", "легкосплав", "r19", "r20", "r21", "r22"]),
        "has_tow_hitch": any(k in low for k in ["фаркоп"]),
        "has_roof_rails": any(k in low for k in ["рейлинг"]),
        "has_memory_seats": any(k in low for k in ["память сиден"]),
        "has_ventilated_seats": any(k in low for k in ["вентиляци сиден"]),
        "has_massage": any(k in low for k in ["массаж"]),
        "has_third_row": any(k in low for k in ["третий ряд", "3-й ряд", "7 мест"]),
        "has_keyless": any(k in low for k in ["бесключевой доступ", "keyless"]),
        "has_apple_carplay": any(k in low for k in ["carplay", "карплей", "android auto"]),
        "has_led_lights": any(k in low for k in ["led", "светодиод", "матричн"]),
        "has_xenon": any(k in low for k in ["ксенон"]),
        "has_multimedia": any(k in low for k in ["мультимедиа", "мультимедий", "экран"]),
        "has_premium_audio": any(k in low for k in ["premium audio", "premium sound", "naim", "burmester", "bose", "harman", "mark levinson"]),
    }

def _extract_condition_flags(description: str | None) -> dict:
    """
    Извлекает флаги состояния из текста описания.
    """
    if not description:
        return {}

    low = description.lower()
    flags = {
        "not_damaged": any(k in low for k in ["не бит", "не крашен", "родная краска"]),
        "one_owner": any(k in low for k in ["один владелец", "1 владелец", "первый владелец"]),
        "dealer_serviced": any(k in low for k in ["у дилера", "официальн", "сервисн"]),
        "no_investments": any(k in low for k in ["не требует вложений", "вложений не требует"]),
        "urgent_sale": any(k in low for k in ["срочно", "срочн", "быстро прода", "быстро продаю"]),
        "exchange": any(k in low for k in ["обмен", "exchange"]),
        "warranty": any(k in low for k in ["гаранти", "warranty"]),
        "no_accidents": any(k in low for k in ["не бит", "не крашен", "родная краска", "без дтп", "без аварий"]),
        "sold_urgently": any(k in low for k in ["срочно", "тороплюсь", "быстро прода"]),
        "negotiable": any(k in low for k in ["торг", "торгуюсь", "уступлю"]),
        "from_dealer": any(k in low for k in ["автосалон", "автомобильный салон", "дилер", "trade-in", "трейд-ин"]),
        "fresh_import": any(k in low for k in ["ввезен", "пригнан", "растаможен", "эптс"]),
        "ev": any(k in low for k in ["электро", "электромобил", "квт", "kwh", "запас хода"]),
    }
    return flags