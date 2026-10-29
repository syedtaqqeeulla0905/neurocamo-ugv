@echo off
TITLE NeuroCamo Tactical SNN Ground Control Station
cd /d "%~dp0"

echo ======================================================================
echo   Launching NeuroCamo Tactical SNN Ground Control Station...
echo ======================================================================

set PYTHON_EXE=%~dp0neurocamo\venv\Scripts\python.exe

"%PYTHON_EXE%" "%~dp0launch_neurocamo_software.py"

pause
