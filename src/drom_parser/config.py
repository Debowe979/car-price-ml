"""Константы для парсера Drom."""

# Имитация браузера — критично для обхода капчи.
# Значение возьми из своего config.py скрейпера (там было IMPERSONATE = ...)
# Типичные значения: "chrome120", "chrome124", "safari17_0"
IMPERSONATE = "chrome131"

# Заголовки HTTP-запроса
HTTP_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}

# Сколько раз пробовать при сетевой ошибке
MAX_RETRIES = 3

# База для экспоненциальной задержки: пауза = RETRY_BACKOFF ** attempt
RETRY_BACKOFF = 2