@echo off
cd /d "%~dp0"

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Set Spark connection (token loaded from .env by Python)
set LLM_SERVER_URL=http://spark-e26c:30000/v1

:: Run MAUDE terminal chat
echo Starting MAUDE Terminal (connecting to Spark at %LLM_SERVER_URL%)
python chat_local.py

pause
