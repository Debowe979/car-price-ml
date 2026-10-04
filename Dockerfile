FROM python:3.12-slim

WORKDIR /app

# системные зависимости для psycopg2
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# сначала только requirements — кешируется отдельно
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# потом весь код
COPY src/ ./src/
COPY models/ ./models/
COPY data/raw/cars.parquet ./data/raw/cars.parquet

# порт API
EXPOSE 8000

# запуск
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]