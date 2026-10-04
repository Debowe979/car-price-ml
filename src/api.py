"""
FastAPI-обёртка над CarPricePredictor.
Запуск: uvicorn src.api:app --host 0.0.0.0 --port 8000
"""
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.db import MODELS_DIR
from src.predict import CarPricePredictor


predictor: Optional[CarPricePredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    print("[startup] loading models...")
    predictor = CarPricePredictor(MODELS_DIR)
    print("[startup] ready")
    yield
    print("[shutdown]")


app = FastAPI(
    title="Car Price Predictor",
    version="1.0",
    lifespan=lifespan,
)


class CarInput(BaseModel):
    # ── основные ─────────────────────────────────────────────
    brand: str = Field(..., example="BMW")
    model: str = Field(..., example="X5")
    year: int = Field(..., ge=1950, le=2030, example=2020)
    generation: Optional[str] = Field(None, example="4 поколение")

    mileage_km: Optional[float] = Field(None, example=80000)
    power_hp: Optional[float] = Field(None, example=340)
    engine_volume_l: Optional[float] = Field(None, example=3.0)
    owners_count: Optional[int] = Field(None, example=2)

    fuel: Optional[str] = Field(None, example="бензин")
    transmission: Optional[str] = Field(None, example="АКПП")
    drive: Optional[str] = Field(None, example="4WD")
    color: Optional[str] = Field(None, example="черный")
    wheel: Optional[str] = Field(None, example="левый")
    seller_type: Optional[str] = Field(None, example="Частное лицо")
    city: Optional[str] = Field(None, example="Москва")
    region: Optional[str] = Field(None, example="Москва")
    seller_description: Optional[str] = Field(None, example="Один владелец, не бит")

    # ── служебные ────────────────────────────────────────────
    photos_count: Optional[int] = Field(0)
    seller_age_years: Optional[int] = Field(0)
    seller_is_new: Optional[bool] = Field(True)

    # ── комплектация (27) ────────────────────────────────────
    opt_has_xenon: bool = False
    opt_has_camera: bool = False
    opt_has_cruise: bool = False
    opt_has_climate: bool = False
    opt_has_keyless: bool = False
    opt_has_leather: bool = False
    opt_has_massage: bool = False
    opt_has_sunroof: bool = False
    opt_has_panorama: bool = False
    opt_has_autopilot: bool = False
    opt_has_bluetooth: bool = False
    opt_has_third_row: bool = False
    opt_has_tow_hitch: bool = False
    opt_has_led_lights: bool = False
    opt_has_multimedia: bool = False
    opt_has_navigation: bool = False
    opt_has_parktronic: bool = False
    opt_has_roof_rails: bool = False
    opt_has_alloy_wheels: bool = False
    opt_has_heated_seats: bool = False
    opt_has_memory_seats: bool = False
    opt_has_apple_carplay: bool = False
    opt_has_premium_audio: bool = False
    opt_has_air_suspension: bool = False
    opt_has_electric_doors: bool = False
    opt_has_heated_steering: bool = False

    # ── состояние (13) ───────────────────────────────────────
    cond_ev: bool = False
    cond_exchange: bool = False
    cond_warranty: bool = False
    cond_one_owner: bool = False
    cond_negotiable: bool = False
    cond_from_dealer: bool = False
    cond_not_damaged: bool = False
    cond_urgent_sale: bool = False
    cond_fresh_import: bool = False
    cond_no_accidents: bool = False
    cond_sold_urgently: bool = False
    cond_no_investments: bool = False
    cond_dealer_serviced: bool = False


class PriceOutput(BaseModel):
    price: float
    range_80_low: float
    range_80_high: float
    range_width_pct: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": predictor is not None}


@app.post("/predict", response_model=PriceOutput)
def predict(car: CarInput):
    if predictor is None:
        raise HTTPException(503, "Model not loaded")

    try:
        result = predictor.predict(car.model_dump())
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")

    low, high = result["range_80"]
    return PriceOutput(
        price=result["price"],
        range_80_low=low,
        range_80_high=high,
        range_width_pct=(high - low) / result["price"] * 100,
    )