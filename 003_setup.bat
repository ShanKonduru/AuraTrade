@echo off
python -m pip install --upgrade pip
@echo off
echo ==============================================
echo AuraTrade - Setup Dependencies
echo ==============================================

echo.
echo Installing Python dependencies...
pip install -r requirements.txt

echo.
echo Creating environment configuration...
if not exist .env (
    copy .env.example .env
    echo Created .env file from template
    echo Please edit .env file with your API keys
) else (
    echo .env file already exists
)

echo.
echo Creating logs directory...
if not exist logs mkdir logs

echo.
echo Setting up data directories...
if not exist data mkdir data
if not exist data\cache mkdir data\cache
if not exist data\models mkdir data\models

echo.
echo Validating installation...
python -c "import sys; print(f'Python version: {sys.version}')"
python -c "import pandas, numpy, yfinance; print('Core data packages: OK')"
python -c "import sklearn, ta; print('Analysis packages: OK')" 2>nul || echo "Some optional packages not installed (will install when needed)"

echo.
echo ==============================================
echo Setup completed successfully!
echo ==============================================
echo.
echo Next steps:
echo 1. Edit .env file with your API keys
echo 2. Run: 004_run.bat
echo.
echo For demo mode (no API keys needed):
echo python main.py --mode demo
echo.
echo For live trading (requires API keys):
echo python main.py --mode trade --symbols AAPL GOOGL MSFT
echo ==============================================

pause
