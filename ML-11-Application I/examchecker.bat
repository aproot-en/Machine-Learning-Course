@echo off
setlocal
cd /d "%~dp0"

set "AEGS_URL=http://127.0.0.1:8000"
set "AEGS_PYTHON=%CD%\.venv\Scripts\python.exe"

if exist "%AEGS_PYTHON%" (
    "%AEGS_PYTHON%" --version >nul 2>&1
    if not errorlevel 1 goto :run
)

where python >nul 2>&1
if errorlevel 1 goto :no_python
set "AEGS_PYTHON=python"

:run
echo Starting AEGS at %AEGS_URL%
echo Close this window or press Ctrl+C to stop the server.
"%AEGS_PYTHON%" main_exam.py --open-browser
if errorlevel 1 goto :failed
goto :end

:no_python
echo Python was not found. Install the project environment first.
pause
exit /b 1

:failed
echo.
echo AEGS could not start. Check the Python environment and dependencies above.
pause
exit /b 1

:end
endlocal
