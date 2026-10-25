import logging
import random
import re
import time

from curl_cffi import requests as curl_requests

from .config import (
    HTTP_HEADERS,
    IMPERSONATE,
    MAX_RETRIES,
    RETRY_BACKOFF,
)

log = logging.getLogger(__name__)


class CaptchaDetected(Exception):
    """Drom вернул капчу или явную блокировку. Скрипт должен остановиться."""
    pass


class HttpClient:
    def __init__(self):
        self.session = curl_requests.Session(impersonate=IMPERSONATE)

    def _looks_like_captcha(self, text: str) -> bool:
        """
        Настоящая капча Drom всегда содержит ключевые слова в теге <title>.
        Проверяем только заголовок, чтобы избежать ложных срабатываний.
        """
        if not text:
            return True

        # Ищем только тег <title>
        m = re.search(r"<title[^>]*>(.*?)</title>", text, re.IGNORECASE | re.DOTALL)
        if m:
            title = m.group(1).lower()
            # Проверяем только ключевые слова в заголовке
            if any(k in title for k in ("капча", "captcha", "доступ ограничен")):
                return True

        return False

    def get_html(self, url: str, max_retries: int = None) -> str | None:
        """
        Скачивает страницу.
        При капче / 403 / 429 — сразу бросает CaptchaDetected, не спит.
        """
        if max_retries is None:
            max_retries = MAX_RETRIES

        last_err = None

        for attempt in range(1, max_retries + 1):
            try:
                r = self.session.get(url, headers=HTTP_HEADERS, timeout=30)
            except Exception as e:
                last_err = e
                log.warning(f"[{attempt}/{max_retries}] сеть упала на {url}: {e}")
                time.sleep(RETRY_BACKOFF ** attempt + random.uniform(0, 1))
                continue

            if r.status_code == 200:
                text = r.text
                if self._looks_like_captcha(text):
                    log.error(f"CAPTCHA на {url}. Останавливаюсь.")
                    raise CaptchaDetected(f"Captcha at {url}")
                return text

            if r.status_code in (403, 429):
                log.error(f"HTTP {r.status_code} на {url}. Блокировка.")
                raise CaptchaDetected(f"HTTP {r.status_code} at {url}")

            if r.status_code == 404:
                log.warning(f"HTTP 404: {url}")
                return None

            log.warning(f"HTTP {r.status_code} на {url}")
            time.sleep(RETRY_BACKOFF ** attempt)

        raise RuntimeError(f"Не удалось скачать {url}: {last_err}")