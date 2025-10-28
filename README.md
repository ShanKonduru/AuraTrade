# Auratrade

# AuraTrade - AI-Enabled Autonomous Trading Platform

**AuraTrade** is a sophisticated multi-agent AI trading system that performs comprehensive fundamental and technical analysis, generates actionable trading signals, and executes trades within predefined risk parameters.

## 🌟 Features

### Multi-Agent Architecture
- **Data Ingestion Agent**: Real-time market data collection and processing
- **Technical Analysis Agent**: Advanced pattern recognition and technical indicators
- **Fundamental Analysis Agent**: DCF valuation and financial health analysis
- **Sentiment Analysis Agent**: News and social media sentiment processing
- **Orchestrator Agent**: Chain-of-Thought reasoning and decision synthesis
- **Execution Agent**: Broker integration and trade execution
- **Risk Management Agent**: Position sizing and portfolio protection

### AI-Powered Analysis
- **LLM Integration**: OpenAI GPT for advanced reasoning and document analysis
- **Machine Learning**: Pattern recognition and predictive modeling
- **Multi-Modal Analysis**: Technical, fundamental, and sentiment data fusion
- **Real-Time Processing**: Continuous market monitoring and signal generation

### Risk Management
- **Position Sizing**: Kelly Criterion and volatility-based algorithms
- **Drawdown Protection**: Real-time monitoring and circuit breakers
- **Correlation Analysis**: Portfolio diversification and risk assessment
- **Risk Limits**: Configurable daily loss and position size limits

### Trading Capabilities
- **Paper Trading**: Risk-free simulation environment
- **Live Trading**: Alpaca broker integration
- **Multi-Asset Support**: Stocks, ETFs, and other securities
- **Order Management**: Advanced order types and execution logic

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Windows environment (batch scripts provided)
- API keys (optional for demo mode)

### Installation

1. **Clone and Setup**
   ```bash
   git clone <repository>
   cd AuraTrade
   ```

2. **Run Setup Script**
   ```bash
   003_setup.bat
   ```
   This will:
   - Install all Python dependencies
   - Create `.env` configuration file
   - Set up data directories
   - Validate installation

3. **Configure API Keys** (Optional for demo)
   Edit `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ALPACA_API_KEY=your_alpaca_api_key_here
   ALPACA_SECRET_KEY=your_alpaca_secret_key_here
   ```

4. **Run AuraTrade**
   ```bash
   004_run.bat
   ```
   Choose from:
   - Demo Mode (no API keys required)
   - Live Trading (requires API keys)
   - Status Check
   - Custom symbols

## 📁 Project Structure

```
AuraTrade/
├── src/
│   ├── agents/                 # Agent implementations
│   │   ├── base_agent.py      # Base agent class
│   │   ├── message_bus.py     # Inter-agent communication
│   │   ├── agent_types.py     # Type definitions
│   │   ├── perception/        # Data ingestion agents
│   │   ├── cognition/         # Analysis agents
│   │   ├── decision/          # Decision-making agents
│   │   └── action/            # Execution agents
│   ├── risk/                  # Risk management
│   ├── auratrade_platform.py  # Main platform orchestrator
│   └── config.py              # Configuration management
├── main.py                    # Application entry point
├── requirements.txt           # Python dependencies
├── .env.example              # Environment template
└── [setup scripts]           # Batch files for setup/run
```

## 🔧 Configuration

### Environment Variables

Key configuration options in `.env`:

**API Keys:**
- `OPENAI_API_KEY`: Required for LLM analysis
- `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`: Required for live trading

**Trading Parameters:**
- `AURA_MAX_POSITION_SIZE=0.25`: Maximum position size (25% of portfolio)
- `AURA_MAX_DRAWDOWN_LIMIT=0.10`: Maximum drawdown (10%)
- `AURA_DAILY_LOSS_LIMIT=0.05`: Daily loss limit (5%)
- `AURA_RISK_PER_TRADE=0.02`: Risk per trade (2%)

**Agent Weights:**
- `AURA_TECHNICAL_WEIGHT=0.35`: Technical analysis weight
- `AURA_FUNDAMENTAL_WEIGHT=0.40`: Fundamental analysis weight
- `AURA_SENTIMENT_WEIGHT=0.25`: Sentiment analysis weight

### Trading Modes

**Demo Mode:**
```bash
python main.py --mode demo
```
- No API keys required
- Shows platform capabilities
- Sample trading signals
- System status monitoring

**Live Trading:**
```bash
python main.py --mode trade --symbols AAPL GOOGL MSFT
```
- Requires API keys
- Real market data
- Actual trade execution
- Continuous monitoring

**Custom Trading:**
```bash
python main.py --mode trade --symbols TSLA NVDA --duration 60
```
- Specific symbols
- Time-limited sessions
- Custom parameters

## 🤖 Agent Details

### Data Ingestion Agent
- **Yahoo Finance**: Primary data source
- **Real-time quotes**: Live market data
- **Historical data**: Backtesting and analysis
- **Caching**: Redis-based response caching
- **Multiple providers**: Extensible data sources

### Technical Analysis Agent
- **Indicators**: RSI, MACD, Bollinger Bands, Moving Averages
- **Pattern Recognition**: Chart patterns and formations
- **ML Models**: Predictive signal generation
- **Timeframe Analysis**: Multi-timeframe signal synthesis

### Fundamental Analysis Agent
- **DCF Valuation**: Discounted Cash Flow models
- **Financial Ratios**: P/E, PEG, debt ratios, profitability metrics
- **LLM Analysis**: Document parsing and qualitative assessment
- **Sector Comparison**: Relative valuation analysis

### Sentiment Analysis Agent
- **News Aggregation**: RSS feeds and financial news
- **NLP Processing**: Multi-model sentiment analysis
- **Event Detection**: Market-moving event identification
- **Social Media**: Sentiment tracking and analysis

### Orchestrator Agent
- **Chain-of-Thought**: LLM-powered reasoning
- **Signal Synthesis**: Multi-agent decision aggregation
- **Conflict Resolution**: Handling contradictory signals
- **Market Regime**: Adaptive strategy selection

### Execution Agent
- **Broker Integration**: Alpaca API support
- **Order Management**: Advanced order types
- **Paper Trading**: Risk-free simulation
- **Portfolio Tracking**: Real-time position monitoring

### Risk Management Agent
- **Position Sizing**: Kelly Criterion optimization
- **Drawdown Monitoring**: Real-time risk assessment
- **Circuit Breakers**: Automated trading halts
- **Correlation Analysis**: Portfolio diversification

## 📊 Usage Examples

### Demo Analysis
```python
# Request trading signal
signal = await platform.request_trading_signal('AAPL')
print(f"Action: {signal['action']}, Confidence: {signal['confidence']}")
```

### System Monitoring
```python
# Check system status
status = await platform.get_system_status()
print(f"Status: {status['status']}, Trading: {status['trading_enabled']}")
```

### Trading Session
```python
# Run trading cycle
results = await platform.run_trading_cycle(['AAPL', 'GOOGL', 'MSFT'])
for symbol, result in results['results'].items():
    print(f"{symbol}: {result}")
```

## ⚠️ Risk Disclaimer

**Important**: This is an experimental trading platform for educational purposes. 

- **Paper Trading Recommended**: Start with paper trading to understand the system
- **Risk Management**: All trading involves risk of loss
- **API Keys**: Keep your API keys secure and never share them
- **No Financial Advice**: This system does not provide financial advice
- **Use at Your Own Risk**: You are responsible for all trading decisions

## 🛠️ Development

### Adding New Agents
1. Inherit from `BaseAgent`
2. Implement `process_message()` method
3. Add to agent initialization in `auratrade_platform.py`
4. Update configuration in `config.py`

### Custom Indicators
1. Add to `technical_analysis_agent.py`
2. Update indicator calculation methods
3. Integrate with signal generation logic

### New Data Sources
1. Extend `data_ingestion_agent.py`
2. Add provider-specific methods
3. Update caching and error handling

## 📝 Logs and Monitoring

- **Application Logs**: `auratrade.log`
- **Detailed Logs**: `logs/auratrade_[timestamp].log`
- **System Status**: Real-time agent health monitoring
- **Trading Results**: Detailed execution logs

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Implement changes with tests
4. Submit pull request
5. Follow code style guidelines

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🔗 Resources

- **Alpaca Trading**: [alpaca.markets](https://alpaca.markets)
- **OpenAI API**: [platform.openai.com](https://platform.openai.com)
- **Technical Analysis**: [TA-Lib documentation](https://mrjbq7.github.io/ta-lib/)
- **Yahoo Finance**: [yfinance documentation](https://pypi.org/project/yfinance/)

---

**AuraTrade** - Where AI meets autonomous trading 🚀

## Batch Files (Windows)

This project includes the following batch files to help with common development tasks on Windows:

* `000_init.bat`: Initialized git and also usn and pwd config setup also done.
* `001_env.bat`: Creates a virtual environment named `venv`.
* `002_activate.bat`: Activates the `venv` virtual environment.
* `003_setup.bat`: Installs the Python packages listed in `requirements.txt` using `pip`.
* `004_run.bat`: Executes the main Python script (`main.py`).
* `005_run_test.bat`: Executes the pytest  scripts (`test_main.py`).
* `005_run_code_cov.bat`: Executes the code coverage pytest  scripts (`test_main.py`).
* `008_deactivate.bat`: Deactivates the currently active virtual environment.

## Contributing

[Explain how others can contribute to your project.]

## License

[Specify the project license, if any.]
