@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set LOGURU_ENCODING=utf-8

start "QueryServer" /B ".\venv\Scripts\python.exe" -m uvicorn app.query_process.api.query_server:app --host 127.0.0.1 --port 8001
start "ImportService" /B ".\venv\Scripts\python.exe" -m uvicorn app.import_process.api.file_import_service:app --host 127.0.0.1 --port 8000
