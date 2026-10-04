@echo off
set PROJECT_DIR=D:\anaconda_projects\car_price_ml
cd /d %PROJECT_DIR%
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set LOG_FILE=%PROJECT_DIR%\reports\retrain_%date:~-4%%date:~3,2%%date:~0,2%.log

echo [%date% %time%] START >> "%LOG_FILE%"
call .venv\Scripts\activate.bat

python -m src.etl >> "%LOG_FILE%" 2>&1
if errorlevel 1 exit /b 1

python -m src.train >> "%LOG_FILE%" 2>&1
if errorlevel 1 exit /b 1

echo [%date% %time%] DONE >> "%LOG_FILE%"