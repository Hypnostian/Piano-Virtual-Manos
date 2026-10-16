@echo off
rem Doble clic para abrir el Piano Virtual con Manos.
rem La primera vez crea el entorno virtual e instala las librerias (necesita internet).
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Preparando el entorno por primera vez, espera unos minutos...
    py -3.11 -m venv .venv 2>nul || python -m venv .venv
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
)
".venv\Scripts\python.exe" app.py %*
if errorlevel 1 pause
