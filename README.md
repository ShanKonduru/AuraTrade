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
- **For local LLM**: [Ollama](https://ollama.ai) (recommended for development)
- **For cloud LLM**: API keys (optional, for production)

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

## 🚀 How to Run AuraTrade

### Method 1: Using Batch Scripts (Windows - Recommended)

**Step 1: Initial Setup**
```bash
.\003_setup.bat
```
This will:
- Install Python dependencies
- Create virtual environment
- Set up configuration files
- Validate installation

**Step 2: Run AuraTrade**
```bash
.\004_run.bat
```
This will show you a menu:
```
1. Demo Mode (no API keys required)
2. Live Trading (requires API keys)
3. Status Check
4. Custom symbols
```

**Choose Option 1 for Demo Mode** - No API keys needed!

### Method 2: Direct Python Commands

**Activate Virtual Environment:**
```bash
.venv\Scripts\activate.bat
```

**Run in Demo Mode:**
```bash
python main_simple.py --mode demo
```

**Check System Status:**
```bash
python main_simple.py --mode status
```

**Live Trading (requires API keys):**
```bash
python main_simple.py --mode trade --symbols AAPL GOOGL MSFT
```

### Method 3: Full System (Advanced)

For the complete multi-agent system with all dependencies:

**Install Additional Dependencies:**
```bash
pip install TA-Lib langchain transformers nltk textblob vaderSentiment
```

**Run Full System:**
```bash
python main.py --mode demo
```

### 🎮 Demo Mode Features

The demo mode shows:
- ✅ Multi-agent workflow simulation
- ✅ Market data collection
- ✅ Technical analysis (RSI, MACD)
- ✅ Fundamental analysis (DCF, ratios)
- ✅ Sentiment analysis (news, social)
- ✅ AI-powered decision making
- ✅ Risk management validation
- ✅ Paper trading execution
- ✅ Real-time progress display

### 🔧 Configuration (Optional)

**For LLM Integration:**
1. Install Ollama: https://ollama.ai
2. Download models: `ollama pull llama3.1:8b`
3. See detailed guide: `docs/OLLAMA_SETUP.md`

**For Live Trading:**
1. Get Alpaca API keys: https://alpaca.markets
2. Update `.env` file:
   ```env
   ALPACA_API_KEY=your_key_here
   ALPACA_SECRET_KEY=your_secret_here
   ```

### 📊 Sample Output

```
🎮 Running AuraTrade in Demo Mode...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Collecting market data...
✅ Market data collected (AAPL, GOOGL, MSFT)

🔍 Running multi-agent analysis...
  🤖 Technical Analysis: Complete
  🤖 Fundamental Analysis: Complete  
  🤖 Sentiment Analysis: Complete

🧠 Orchestrator making decisions...
  🎯 AAPL: BUY signal (Confidence: 85%)
  🎯 GOOGL: HOLD signal (Confidence: 65%)
  🎯 MSFT: BUY signal (Confidence: 78%)

🛡️ Risk management validation...
  ✅ Position sizing: Within limits
  
⚡ Executing trades (Paper Trading)...
  📈 BUY AAPL: 10 shares @ $178.50
  
🎉 Demo trading session complete!
📊 Portfolio value: $10,000 → $10,150 (+1.5%)
```

### 🎯 Quick Start Summary

1. **Clone** the repository
2. **Run** `.\003_setup.bat` (Windows) 
3. **Run** `.\004_run.bat` and choose **Option 1**
4. **Enjoy** the demo! 🚀

No API keys needed for demo mode!

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

## 🏗️ System Architecture

AuraTrade implements a sophisticated multi-agent architecture with flexible LLM integration. The system is designed for scalability, maintainability, and cost-effective development.

### High-Level Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           AuraTrade Trading Platform                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│  External Data → Data Ingestion → Analysis Agents → Orchestrator → Execution   │
│                                                         │                       │
│                                                         ▼                       │
│                                                   Risk Management               │
│                                                         │                       │
│                                                         ▼                       │
│                                                    Final Trade                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Agent Communication Flow

```text
┌─────────────┐    ┌─────────────────┐    ┌─────────────────────────────────┐
│ Market Data │───▶│ Data Ingestion  │───▶│        Analysis Layer           │
│ News Feeds  │    │     Agent       │    │                                 │
│ Financial   │    │                 │    │ ┌─────────────┐ ┌─────────────┐ │
│ Reports     │    │ • Data Cleaning │    │ │ Technical   │ │ Fundamental │ │
└─────────────┘    │ • Caching       │    │ │ Analysis    │ │ Analysis    │ │
                   │ • Validation    │    │ │             │ │             │ │
                   └─────────────────┘    │ │ • RSI/MACD  │ │ • DCF Model │ │
                                          │ │ • Patterns  │ │ • Ratios    │ │
                                          │ │ • ML Signals│ │ • LLM Docs  │ │
                                          │ └─────────────┘ └─────────────┘ │
                                          │                                 │
                                          │ ┌─────────────┐                 │
                                          │ │ Sentiment   │                 │
                                          │ │ Analysis    │                 │
                                          │ │             │                 │
                                          │ │ • News NLP  │                 │
                                          │ │ • Social    │                 │
                                          │ │ • Events    │                 │
                                          │ └─────────────┘                 │
                                          └─────────────────────────────────┘
                                                          │
                                                          ▼
                                          ┌─────────────────────────────────┐
                                          │        Orchestrator Agent       │
                                          │                                 │
                                          │ • Chain-of-Thought Reasoning    │
                                          │ • Signal Aggregation            │
                                          │ • Conflict Resolution           │
                                          │ • Confidence Scoring            │
                                          │ • Action Recommendation         │
                                          └─────────────────────────────────┘
                                                          │
                                                          ▼
                                          ┌─────────────────────────────────┐
                                          │      Risk Management Agent      │
                                          │                                 │
                                          │ • Position Sizing               │
                                          │ • Drawdown Monitoring           │
                                          │ • Risk Limits Check             │
                                          │ • Correlation Analysis          │
                                          │ • Circuit Breakers              │
                                          └─────────────────────────────────┘
                                                          │
                                                          ▼
                                          ┌─────────────────────────────────┐
                                          │       Execution Agent           │
                                          │                                 │
                                          │ • Order Management              │
                                          │ • Broker Integration            │
                                          │ • Trade Execution               │
                                          │ • Portfolio Tracking            │
                                          │ • Paper/Live Trading            │
                                          └─────────────────────────────────┘
```

### LLM Provider Architecture

AuraTrade features a flexible LLM provider system that enables cost-effective development with local models and reliable production deployment with cloud APIs:

```text
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            LLM Manager                                         │
│                                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐            │
│  │   Primary LLM   │    │  Fallback LLMs  │    │  Health Monitor │            │
│  │                 │    │                 │    │                 │            │
│  │ ┌─────────────┐ │    │ ┌─────────────┐ │    │ • Availability  │            │
│  │ │   Ollama    │ │    │ │   OpenAI    │ │    │ • Response Time │            │
│  │ │  (Local)    │ │    │ │  (Cloud)    │ │    │ • Error Rate    │            │
│  │ │             │ │    │ │             │ │    │ • Auto-Failover │            │
│  │ │ • llama3.1  │ │    │ │ • gpt-3.5   │ │    └─────────────────┘            │
│  │ │ • Free      │ │    │ │ • Reliable  │ │                                   │
│  │ │ • Private   │ │    │ │ • Fast      │ │    ┌─────────────────┐            │
│  │ │ • Fast      │ │    │ └─────────────┘ │    │ Request Router  │            │
│  │ └─────────────┘ │    │                 │    │                 │            │
│  └─────────────────┘    │ ┌─────────────┐ │    │ • Load Balance  │            │
│                         │ │ Anthropic   │ │    │ • Retry Logic   │            │
│                         │ │  Claude     │ │    │ • Error Handle  │            │
│                         │ │             │ │    │ • Cost Optimize │            │
│                         │ │ • Advanced  │ │    └─────────────────┘            │
│                         │ │ • Accurate  │ │                                   │
│                         │ └─────────────┘ │                                   │
│                         └─────────────────┘                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Development Mode**: Use local Ollama for free LLM capabilities  
**Production Mode**: Automatic failover to OpenAI/Anthropic for reliability

### Quick Architecture Overview

**Trading Decision Flow:**
```
Market Data → Data Ingestion → Analysis (Technical + Fundamental + Sentiment) 
                    ↓
     Orchestrator (AI Reasoning) → Risk Management → Trade Execution
```

**Agent Types:**
- **Perception**: Data Ingestion Agent
- **Cognition**: Technical Analysis, Fundamental Analysis, Sentiment Analysis Agents  
- **Decision**: Orchestrator Agent (Chain-of-Thought reasoning)
- **Action**: Execution Agent
- **Safety**: Risk Management Agent

**LLM Integration:**
- **Local Development**: Ollama (free, private, fast)
- **Production**: OpenAI/Anthropic with automatic failover
- **Used for**: Document analysis, reasoning, decision synthesis

For detailed architecture diagrams and technical specifications, see [System Architecture Documentation](docs/SYSTEM_ARCHITECTURE.md).

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
