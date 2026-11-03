@echo off
REM Starts the RefillCare Streamlit app with the project environment (.venv: Python 3.13, scikit-learn 1.7.2).
REM Do not use the system Python 3.9 "streamlit" - it cannot load the ML model.
cd /d "%~dp0"
if not exist ".venv\Scripts\streamlit.exe" (
  echo .venv not found. Create it: py -3.13 -m venv .venv ^&^& .venv\Scripts\pip install -r requirements.txt
  pause
  exit /b 1
)
".venv\Scripts\streamlit.exe" run app_refillcare.py %*
