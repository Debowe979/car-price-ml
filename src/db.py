"""
Пути проекта + подключение к PostgreSQL.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

# ── пути ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent   # drom_ml/

DATA_DIR       = PROJECT_ROOT / "data"
DATA_RAW       = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
MODELS_DIR     = PROJECT_ROOT / "models"
REPORTS_DIR    = PROJECT_ROOT / "reports"

for d in (DATA_RAW, DATA_PROCESSED, MODELS_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ── доступ к БД ───────────────────────────────────────────────────────

load_dotenv(PROJECT_ROOT / ".env")   # .env в корне проекта


def get_engine() -> Engine:
    """SQLAlchemy-движок к PostgreSQL. Читает креды из .env."""
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASS")
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME")

    if not all([user, password, name]):
        raise RuntimeError(
            "Не заданы DB_USER / DB_PASS / DB_NAME. "
            f"Проверь {PROJECT_ROOT / '.env'}"
        )

    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"
    return create_engine(url, pool_pre_ping=True)


def get_connection():
    """Сырое соединение, если нужно psycopg2 напрямую."""
    return get_engine().raw_connection()