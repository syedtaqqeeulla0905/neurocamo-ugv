@echo off
REM Launch NeuroCamo Web on http://127.0.0.1:8080 using the neurocamo venv.
setlocal
set HERE=%~dp0
call "%HERE%..\neurocamo\venv\Scripts\python.exe" "%HERE%app.py"
endlocal
