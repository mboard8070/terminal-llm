@echo off
cd /d "%~dp0"

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Set Ubuntu MAUDE server connection (token loaded from .env by Python)
set LLM_SERVER_URL=http://server:30080/v1

:: Run MAUDE terminal chat
echo Starting MAUDE Terminal (connecting to server at %LLM_SERVER_URL%)
python chat_local.py

pause
