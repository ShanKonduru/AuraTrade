@echo off
echo ==============================================
echo AuraTrade - AI Autonomous Trading Platform
echo ==============================================

echo.
echo Choose run mode:
echo 1. Demo Mode (no API keys required)
echo 2. Live Trading (requires API keys)
echo 3. Status Check
echo 4. Custom symbols
echo.

set /p choice="Enter choice (1-4): "

if "%choice%"=="1" (
    echo.
    echo Starting AuraTrade in Demo Mode...
    echo This will show platform capabilities without live trading
    echo.
    .venv\Scripts\python.exe main_simple.py --mode demo
) else if "%choice%"=="2" (
    echo.
    echo Starting AuraTrade in Live Trading Mode...
    echo This will perform actual trading using configured API keys
    echo Press Ctrl+C to stop
    echo.
    .venv\Scripts\python.exe main_simple.py --mode trade
) else if "%choice%"=="3" (
    echo.
    echo Checking AuraTrade Status...
    echo.
    .venv\Scripts\python.exe main_simple.py --mode status
) else if "%choice%"=="4" (
    echo.
    set /p symbols="Enter symbols (space-separated, e.g., AAPL GOOGL TSLA): "
    set /p duration="Enter duration in minutes (leave empty for continuous): "
    
    if "%duration%"=="" (
        echo Starting trading for symbols: %symbols%
        .venv\Scripts\python.exe main_simple.py --mode trade --symbols %symbols%
    ) else (
        echo Starting trading for symbols: %symbols% for %duration% minutes
        .venv\Scripts\python.exe main_simple.py --mode trade --symbols %symbols% --duration %duration%
    )
) else (
    echo Invalid choice. Please run again and select 1-4.
)

echo.
echo ==============================================
pause
