@echo off
setlocal

title ML 3D Simulation
cd /d "%~dp0"

where node >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js is not installed or is not available in PATH.
    echo Download Node.js from https://nodejs.org/ and try again.
    pause
    exit /b 1
)

where npm >nul 2>&1
if errorlevel 1 (
    echo [ERROR] npm is not installed or is not available in PATH.
    pause
    exit /b 1
)

if not exist "node_modules\vite\bin\vite.js" (
    echo Installing dependencies...
    call npm install
    if errorlevel 1 (
        echo.
        echo [ERROR] Dependency installation failed.
        pause
        exit /b 1
    )
)

echo Starting ML 3D Simulation...
echo The web browser will open automatically.
echo Press Ctrl+C to stop the server.
echo.

call npm run dev -- --open --host 127.0.0.1

if errorlevel 1 (
    echo.
    echo [ERROR] The web server stopped with an error.
    pause
    exit /b 1
)

endlocal
