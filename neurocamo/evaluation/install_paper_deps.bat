@echo off
cd /d E:\major_project\neurocamo
echo Installing paper-grade evaluation dependencies into the venv...
.\venv\Scripts\python.exe -m pip install torchmetrics scikit-learn captum pandas
echo.
echo Done. Now run:  venv\Scripts\python.exe evaluation\paper_eval.py
pause
