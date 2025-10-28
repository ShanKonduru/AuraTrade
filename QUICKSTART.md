# 🚀 Quick Start Guide - AuraTrade AI Trading Platform

## ⚡ **1-Minute Setup**

### **Prerequisites**
- Python 3.8+ installed
- Git installed
- Internet connection

### **Quick Launch**
```bash
# Clone the repository
git clone https://github.com/yourusername/AuraTrade.git
cd AuraTrade

# Install dependencies  
pip install -r requirements.txt

# Launch the platform
python launch_auratrade.py
```

That's it! 🎉 The platform will automatically:
- Start the MCP server at `http://localhost:8000`
- Start the chatbot interface at `http://localhost:8080` 
- Open your browser to the trading assistant

---

## 🤖 **Using the Trading Assistant**

### **Example Conversations**

**📊 Stock Analysis**
```
You: "Analyze Apple stock"
Bot: Provides comprehensive AAPL analysis with price, trends, and recommendations
```

**💰 Trading Advice**
```
You: "Should I buy Tesla with $5000?"
Bot: Gives buy/sell recommendation with entry/exit prices and risk analysis
```

**🛡️ Risk Management**
```
You: "Calculate stop loss for my Microsoft position"
Bot: Provides stop-loss levels and risk calculations
```

**📈 Stock Discovery**
```
You: "Show me good growth stocks"
Bot: Lists growth stock categories with specific recommendations
```

### **Quick Action Buttons**
- 📊 **Analyze AAPL** - Get Apple stock analysis
- 📈 **Growth Stocks** - See growth stock recommendations  
- 💰 **Dividend Stocks** - Find dividend-paying stocks
- 📰 **Market News** - Get current market overview

---

## 🔧 **Advanced Usage**

### **API Access**
The MCP server provides REST API access at `http://localhost:8000`

**Example API Calls:**
```python
import requests

# Market analysis
response = requests.post("http://localhost:8000/mcp", json={
    "jsonrpc": "2.0",
    "id": "1",
    "method": "analyze", 
    "params": {"symbol": "AAPL"}
})

# Trading recommendation
response = requests.post("http://localhost:8000/mcp", json={
    "jsonrpc": "2.0", 
    "id": "2",
    "method": "recommend",
    "params": {"symbol": "TSLA", "amount": 5000}
})
```

### **Configuration**
The platform runs in demo mode by default with simulated data. To use real market data:

1. Configure API keys in environment variables
2. Update `src/config.py` with real data sources
3. Restart the platform

---

## 🎯 **Features Overview**

### **🤖 AI Trading Chatbot**
- Natural language trading queries
- Real-time conversation interface
- Context-aware responses
- Educational trading content

### **📊 Market Analysis**
- Technical indicator analysis
- Fundamental company analysis  
- Market sentiment analysis
- Multi-timeframe support

### **🎯 Trading Recommendations**
- Buy/sell/hold recommendations
- Entry and exit price targets
- Risk-reward ratio calculations
- Position sizing guidance

### **🛡️ Risk Management**
- Stop-loss calculations
- Portfolio risk assessment
- Value at Risk (VaR) analysis
- Diversification scoring

### **🔗 MCP Server Integration**
- Model Context Protocol compliance
- REST API for external integration
- WebSocket real-time data streaming
- Extensible plugin architecture

---

## 📱 **Platform Architecture**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Browser   │◄──►│  Chatbot UI     │◄──►│  MCP Server     │
│  (Port 8080)    │    │  (FastAPI)      │    │  (Port 8000)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │ Trading Agents  │
                                               │ • Data Ingestion│
                                               │ • Technical     │
                                               │ • Fundamental   │
                                               │ • Sentiment     │
                                               │ • Risk Mgmt     │
                                               │ • Execution     │
                                               └─────────────────┘
```

---

## 🛠️ **Troubleshooting**

### **Common Issues**

**Port Already in Use**
```bash
# Check what's using the port
netstat -ano | findstr :8000
netstat -ano | findstr :8080

# Kill the process if needed
taskkill /PID <process_id> /F
```

**Dependencies Issues** 
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Or use virtual environment
python -m venv auratrade_env
auratrade_env\Scripts\activate
pip install -r requirements.txt
```

**Can't Connect to MCP Server**
- Check if MCP server is running at `http://localhost:8000/health`
- Verify firewall settings
- Check console for error messages

### **Getting Help**
- 📧 Check the logs in the console output
- 🐛 Report issues on GitHub
- 📖 See `project_plan.md` for detailed documentation
- 💬 Use the chatbot to ask questions about the system

---

## 🚀 **What's Next?**

The platform includes:
- ✅ **Complete trading intelligence system**
- ✅ **AI-powered chatbot interface** 
- ✅ **MCP server for integrations**
- ✅ **Risk management tools**
- ✅ **Real-time market analysis**

**Try asking the chatbot:**
- "What's your analysis on Apple?"
- "Should I invest in Tesla with $10,000?"
- "What are some good dividend stocks?"
- "Calculate stop loss for Microsoft"
- "What's happening in the market today?"

Happy trading! 📈✨