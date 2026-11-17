@echo off
REM One-time setup (Windows). Usage: scripts\setup.bat [--ml]
cd /d "%~dp0\.."
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r backend\requirements.txt -r frontend\requirements.txt
if "%1"=="--ml" pip install -r backend\requirements-ml.txt
if not exist .env copy .env.example .env
echo Done. Activate with .venv\Scripts\activate  then run: python run.py
