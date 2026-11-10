@echo off
cd /d "%~dp0"
python -m venv .venv
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements-app.txt
if errorlevel 1 goto failed
echo Setup complete. Double-click Start_TrialScreen.cmd to open the browser.
pause
exit /b 0
:failed
echo Setup failed. Read the error above before retrying.
pause
exit /b 1
