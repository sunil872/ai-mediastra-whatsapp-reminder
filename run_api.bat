@echo off
REM Starts the RefillCare FastAPI server with the project environment (.venv).
cd /d "%~dp0"
".venv\Scripts\python.exe" -m uvicorn api.main:app --host 127.0.0.1 --port 8000 %*
