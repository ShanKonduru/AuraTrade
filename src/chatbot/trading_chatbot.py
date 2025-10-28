"""
AuraTrade Trading Chatbot
Natural language interface for trading intelligence
"""

import asyncio
import re
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import aiohttp
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

@dataclass
class ChatMessage:
    """Chat message structure"""
    content: str
    timestamp: datetime
    is_user: bool
    intent: Optional[str] = None
    entities: Optional[Dict[str, Any]] = None

@dataclass
class Intent:
    """Recognized user intent"""
    name: str
    confidence: float
    entities: Dict[str, Any]

class SimpleNLPProcessor:
    """Simple NLP processor for trading queries"""
    
    def __init__(self):
        self.intent_patterns = {
            "market_analysis": [
                r"analy[sz]e?\s+(\w+)",
                r"what.*about\s+(\w+)",
                r"tell me about\s+(\w+)",
                r"(\w+)\s+analysis",
                r"how.*(\w+)\s+doing",
                r"(\w+)\s+performance"
            ],
            "trading_recommendation": [
                r"should.*buy\s+(\w+)",
                r"should.*sell\s+(\w+)",
                r"recommend.*(\w+)",
                r"(\w+)\s+recommendation",
                r"invest.*(\w+)",
                r"trade.*(\w+)"
            ],
            "price_inquiry": [
                r"price.*(\w+)",
                r"(\w+)\s+price",
                r"cost.*(\w+)",
                r"how much.*(\w+)"
            ],
            "stop_loss": [
                r"stop loss.*(\w+)",
                r"(\w+)\s+stop loss",
                r"risk.*(\w+)",
                r"set stop.*(\w+)"
            ],
            "portfolio_risk": [
                r"portfolio risk",
                r"risk.*portfolio",
                r"my risk",
                r"portfolio.*safe"
            ],
            "growth_stocks": [
                r"growth stocks",
                r"growing.*stocks",
                r"best.*growth",
                r"stocks.*growth"
            ],
            "dividend_stocks": [
                r"dividend stocks",
                r"dividend.*paying",
                r"income.*stocks",
                r"yield.*stocks"
            ],
            "market_news": [
                r"market news",
                r"news.*market",
                r"what.*happening",
                r"market.*today"
            ]
        }
        
        self.symbol_pattern = r"\b([A-Z]{1,5})\b"
        self.amount_pattern = r"\$?(\d+(?:,\d{3})*(?:\.\d{2})?)"
        
    def extract_intent(self, text: str) -> Intent:
        """Extract intent from user input"""
        text_lower = text.lower()
        
        for intent_name, patterns in self.intent_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    entities = self._extract_entities(text, match)
                    return Intent(
                        name=intent_name,
                        confidence=0.8,
                        entities=entities
                    )
        
        return Intent(
            name="unknown",
            confidence=0.1,
            entities={}
        )
    
    def _extract_entities(self, text: str, intent_match: re.Match) -> Dict[str, Any]:
        """Extract entities from text"""
        entities = {}
        
        # Extract stock symbols
        symbols = re.findall(self.symbol_pattern, text.upper())
        if symbols:
            entities["symbols"] = symbols
        
        # Extract amounts
        amounts = re.findall(self.amount_pattern, text)
        if amounts:
            entities["amounts"] = [float(amt.replace(",", "")) for amt in amounts]
        
        # Extract symbol from intent match if available
        if intent_match and intent_match.groups():
            symbol = intent_match.group(1).upper()
            if len(symbol) <= 5 and symbol.isalpha():
                entities["primary_symbol"] = symbol
        
        return entities

class MCPClient:
    """MCP client for communicating with AuraTrade MCP server"""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send MCP request to server"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        request_data = {
            "jsonrpc": "2.0",
            "id": str(datetime.now().timestamp()),
            "method": method,
            "params": params
        }
        
        try:
            async with self.session.post(f"{self.server_url}/mcp", json=request_data) as response:
                result = await response.json()
                
                if "error" in result:
                    raise Exception(f"MCP Error: {result['error']['message']}")
                
                return result.get("result", {})
                
        except aiohttp.ClientError as e:
            raise Exception(f"Connection error: {str(e)}")
        except Exception as e:
            raise Exception(f"Request failed: {str(e)}")

class TradingChatbot:
    """AI Trading Assistant Chatbot"""
    
    def __init__(self, mcp_server_url: str = "http://localhost:8000"):
        self.nlp = SimpleNLPProcessor()
        self.mcp_server_url = mcp_server_url
        self.conversation_history: List[ChatMessage] = []
        self.user_context: Dict[str, Any] = {}
    
    async def process_message(self, user_input: str) -> str:
        """Process user message and return bot response"""
        # Store user message
        user_msg = ChatMessage(
            content=user_input,
            timestamp=datetime.now(),
            is_user=True
        )
        self.conversation_history.append(user_msg)
        
        # Extract intent and entities
        intent = self.nlp.extract_intent(user_input)
        user_msg.intent = intent.name
        user_msg.entities = intent.entities
        
        # Generate response based on intent
        try:
            response = await self._handle_intent(intent, user_input)
        except Exception as e:
            response = f"❌ Sorry, I encountered an error: {str(e)}\n\n💡 Please try again or rephrase your question."
        
        # Store bot response
        bot_msg = ChatMessage(
            content=response,
            timestamp=datetime.now(),
            is_user=False
        )
        self.conversation_history.append(bot_msg)
        
        return response
    
    async def _handle_intent(self, intent: Intent, user_input: str) -> str:
        """Handle specific user intent"""
        
        if intent.name == "market_analysis":
            return await self._handle_market_analysis(intent)
        
        elif intent.name == "trading_recommendation":
            return await self._handle_trading_recommendation(intent)
        
        elif intent.name == "price_inquiry":
            return await self._handle_price_inquiry(intent)
        
        elif intent.name == "stop_loss":
            return await self._handle_stop_loss(intent)
        
        elif intent.name == "portfolio_risk":
            return await self._handle_portfolio_risk(intent)
        
        elif intent.name == "growth_stocks":
            return await self._handle_growth_stocks(intent)
        
        elif intent.name == "dividend_stocks":
            return await self._handle_dividend_stocks(intent)
        
        elif intent.name == "market_news":
            return await self._handle_market_news(intent)
        
        else:
            return self._handle_unknown_intent(user_input)
    
    async def _handle_market_analysis(self, intent: Intent) -> str:
        """Handle market analysis requests"""
        symbol = intent.entities.get("primary_symbol") or \
                intent.entities.get("symbols", [None])[0]
        
        if not symbol:
            return "🤔 Which stock would you like me to analyze? Please provide a stock symbol (e.g., AAPL, GOOGL, MSFT)."
        
        async with MCPClient(self.mcp_server_url) as client:
            result = await client.send_request("analyze", {
                "symbol": symbol,
                "timeframe": "1d",
                "include_technical": True,
                "include_fundamental": True,
                "include_sentiment": True
            })
        
        if "error" in result:
            return f"❌ Unable to analyze {symbol}: {result['error']}"
        
        return self._format_market_analysis(symbol, result)
    
    async def _handle_trading_recommendation(self, intent: Intent) -> str:
        """Handle trading recommendation requests"""
        symbol = intent.entities.get("primary_symbol") or \
                intent.entities.get("symbols", [None])[0]
        amount = intent.entities.get("amounts", [None])[0]
        
        if not symbol:
            return "🤔 Which stock are you considering? Please provide a stock symbol."
        
        async with MCPClient(self.mcp_server_url) as client:
            params = {"symbol": symbol}
            if amount:
                params["amount"] = amount
            
            result = await client.send_request("recommend", params)
        
        if "error" in result:
            return f"❌ Unable to get recommendation for {symbol}: {result['error']}"
        
        return self._format_trading_recommendation(symbol, result, amount)
    
    async def _handle_price_inquiry(self, intent: Intent) -> str:
        """Handle price inquiry requests"""
        symbol = intent.entities.get("primary_symbol") or \
                intent.entities.get("symbols", [None])[0]
        
        if not symbol:
            return "🤔 Which stock price would you like to know? Please provide a stock symbol."
        
        async with MCPClient(self.mcp_server_url) as client:
            result = await client.send_request("analyze", {
                "symbol": symbol,
                "include_technical": False,
                "include_fundamental": False,
                "include_sentiment": False
            })
        
        if "error" in result:
            return f"❌ Unable to get price for {symbol}: {result['error']}"
        
        market_data = result.get("market_data", {})
        price = market_data.get("price", 0)
        change = market_data.get("change", 0)
        change_sign = "+" if change >= 0 else ""
        
        return f"💰 {symbol} Current Price: ${price:.2f} ({change_sign}{change:.2f})"
    
    async def _handle_stop_loss(self, intent: Intent) -> str:
        """Handle stop loss calculation requests"""
        symbol = intent.entities.get("primary_symbol") or \
                intent.entities.get("symbols", [None])[0]
        
        if not symbol:
            return "🤔 For which stock would you like stop-loss guidance? Please provide more details about your position."
        
        # For demo, use current price as entry price
        async with MCPClient(self.mcp_server_url) as client:
            # Get current price first
            analysis = await client.send_request("analyze", {
                "symbol": symbol,
                "include_technical": False,
                "include_fundamental": False,
                "include_sentiment": False
            })
            
            current_price = analysis.get("market_data", {}).get("price", 100)
            
            # Calculate stop loss
            result = await client.send_request("stop_loss", {
                "symbol": symbol,
                "entry_price": current_price,
                "position_size": 100,  # Default 100 shares
                "risk_percentage": 2.0
            })
        
        if "error" in result:
            return f"❌ Unable to calculate stop-loss for {symbol}: {result['error']}"
        
        return self._format_stop_loss(symbol, result)
    
    async def _handle_portfolio_risk(self, intent: Intent) -> str:
        """Handle portfolio risk assessment"""
        # For demo, return general risk guidance
        return """🛡️ **Portfolio Risk Assessment**

**Key Risk Metrics to Monitor:**
📊 **Diversification**: Spread across 8-12 different stocks
🏢 **Sector Allocation**: No more than 40% in one sector
📈 **Beta**: Portfolio beta between 0.8-1.2 for balanced risk
📉 **Max Drawdown**: Keep historical losses under 15%

**Risk Management Tips:**
• Position sizing: No single stock > 10% of portfolio
• Stop losses: Set at 7-10% below entry price
• Regular rebalancing: Monthly or quarterly
• Cash reserves: Keep 10-20% in cash/bonds

💡 **Want specific analysis?** Share your current positions and I'll analyze your portfolio risk!"""
    
    async def _handle_growth_stocks(self, intent: Intent) -> str:
        """Handle growth stocks recommendations"""
        return """🚀 **Top Growth Stock Categories to Consider:**

**Technology Leaders:**
• **Cloud Computing**: CRM, SNOW, DDOG
• **AI/Semiconductors**: NVDA, AMD, TSM
• **Software**: MSFT, GOOGL, META

**Emerging Sectors:**
• **Clean Energy**: TSLA, ENPH, PLUG
• **Biotech**: MRNA, GILD, BIIB
• **Fintech**: SQ, PYPL, AFRM

**Screening Criteria:**
📈 Revenue growth > 20% annually
💰 Strong cash flow and profitability
🎯 Expanding market opportunity
👥 Strong management team

💡 **Want specific analysis?** Ask me to "analyze [SYMBOL]" for any of these stocks!"""
    
    async def _handle_dividend_stocks(self, intent: Intent) -> str:
        """Handle dividend stocks recommendations"""
        return """💰 **Quality Dividend Stock Categories:**

**Dividend Aristocrats (25+ years of increases):**
• **Consumer Staples**: KO, PG, WMT
• **Utilities**: NEE, SO, DUK
• **REITs**: O, SPG, PLD

**High-Yield Opportunities:**
• **Telecom**: VZ, T
• **Energy**: XOM, CVX, ENB
• **Finance**: JPM, BAC, WFC

**Dividend Quality Metrics:**
📊 Yield: 3-6% (sustainable range)
💪 Payout Ratio: <60% of earnings
📈 Growth: 5-10% annual increases
🏛️ Stability: Consistent through cycles

💡 **Want specific analysis?** Ask me about any dividend stock for detailed metrics!"""
    
    async def _handle_market_news(self, intent: Intent) -> str:
        """Handle market news requests"""
        return f"""📰 **Current Market Overview:**

**Market Sentiment**: Mixed signals with rotation between sectors
**Key Themes**:
• Federal Reserve policy impacts
• Earnings season results
• Tech sector volatility
• Energy and commodity trends

**📊 Major Indices Today:**
• S&P 500: Monitoring key support levels
• NASDAQ: Tech earnings driving movement  
• Dow Jones: Value rotation continues

**🔍 Sectors to Watch:**
• Technology: AI and cloud companies
• Healthcare: Biotech developments
• Energy: Oil price movements
• Financials: Interest rate sensitivity

💡 **Want specific stock analysis?** Ask me about any symbol for detailed insights!

*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*"""
    
    def _handle_unknown_intent(self, user_input: str) -> str:
        """Handle unknown or unclear intents"""
        return """🤔 **I'm here to help with trading questions!**

**I can assist you with:**

📊 **Market Analysis**: "Analyze AAPL" or "What about Tesla?"
🎯 **Trading Advice**: "Should I buy MSFT?" or "GOOGL recommendation?"
💰 **Price Checks**: "AMZN price" or "How much is Netflix?"
🛡️ **Risk Management**: "Stop loss for my TSLA position"
📈 **Stock Screening**: "Show me growth stocks" or "Dividend stocks"
📰 **Market News**: "What's happening in the market?"

**Example questions:**
• "What's your analysis on Apple?"
• "Should I invest in Tesla with $5000?"
• "What are some good dividend stocks?"
• "Calculate stop loss for Microsoft"

💡 **Try asking about any stock symbol or trading topic!**"""
    
    def _format_market_analysis(self, symbol: str, result: Dict[str, Any]) -> str:
        """Format market analysis results"""
        market_data = result.get("market_data", {})
        technical = result.get("technical", {})
        fundamental = result.get("fundamental", {})
        sentiment = result.get("sentiment", {})
        
        price = market_data.get("price", 0)
        change = market_data.get("change", 0)
        change_sign = "+" if change >= 0 else ""
        
        response = f"""📊 **{symbol} Market Analysis**

**💰 Current Price**: ${price:.2f} ({change_sign}{change:.2f})
**📅 Analysis Time**: {datetime.now().strftime('%Y-%m-%d %H:%M')}

"""
        
        if technical:
            trend = technical.get("trend", "neutral").title()
            strength = technical.get("strength", 0.5)
            response += f"""**📈 Technical Analysis**:
• Trend: {trend} (Strength: {strength*100:.0f}%)
• Signals: {', '.join(technical.get('signals', ['No clear signals']))}

"""
        
        if fundamental:
            score = fundamental.get("score", 0.5)
            recommendation = fundamental.get("recommendation", "hold").upper()
            response += f"""**📋 Fundamental Analysis**:
• Overall Score: {score*100:.0f}%
• Recommendation: {recommendation}

"""
        
        if sentiment:
            sent_score = sentiment.get("score", 0.0)
            confidence = sentiment.get("confidence", 0.5)
            sent_text = "Positive" if sent_score > 0.1 else "Negative" if sent_score < -0.1 else "Neutral"
            response += f"""**🗞️ Sentiment Analysis**:
• Market Sentiment: {sent_text} ({sent_score:+.2f})
• Confidence: {confidence*100:.0f}%

"""
        
        response += "💡 **Want more details?** Ask me for specific recommendations or risk analysis!"
        
        return response
    
    def _format_trading_recommendation(self, symbol: str, result: Dict[str, Any], amount: Optional[float]) -> str:
        """Format trading recommendation"""
        action = result.get("recommendation", "hold").upper()
        confidence = result.get("confidence", 0.5)
        reasoning = result.get("reasoning", "Based on current market analysis")
        entry_price = result.get("entry_price")
        target_price = result.get("target_price")
        stop_loss = result.get("stop_loss")
        
        # Action emoji
        action_emoji = {
            "BUY": "📈",
            "SELL": "📉", 
            "HOLD": "⏸️",
            "ACCUMULATE": "📊",
            "DIVEST": "📉"
        }.get(action, "🤔")
        
        response = f"""🎯 **{symbol} Trading Recommendation**

**{action_emoji} Action**: {action}
**📊 Confidence**: {confidence*100:.0f}%

**💭 Reasoning**: {reasoning}

"""
        
        if entry_price:
            response += f"**💰 Entry Price**: ${entry_price:.2f}\n"
        
        if target_price:
            response += f"**🎯 Target Price**: ${target_price:.2f}\n"
        
        if stop_loss:
            response += f"**🛡️ Stop Loss**: ${stop_loss:.2f}\n"
        
        if amount and entry_price:
            shares = int(amount / entry_price)
            response += f"\n**📊 Position Sizing** (for ${amount:,.0f}):\n"
            response += f"• Shares: {shares}\n"
            response += f"• Investment: ${shares * entry_price:,.2f}\n"
        
        response += "\n💡 **Remember**: This is analysis, not financial advice. Always do your own research!"
        
        return response
    
    def _format_stop_loss(self, symbol: str, result: Dict[str, Any]) -> str:
        """Format stop loss calculation"""
        entry_price = result.get("entry_price", 0)
        stop_loss = result.get("recommended_stop_loss", 0)
        risk_pct = result.get("risk_percentage", 0)
        rationale = result.get("rationale", "")
        
        return f"""🛡️ **{symbol} Stop Loss Calculation**

**💰 Entry Price**: ${entry_price:.2f}
**🚨 Recommended Stop**: ${stop_loss:.2f}
**📉 Risk Amount**: {risk_pct:.1f}%

**🧠 Rationale**: {rationale}

**⚡ Quick Setup**:
1. Set stop-loss order at ${stop_loss:.2f}
2. Monitor for any breakdown
3. Consider trailing stop as price rises

💡 **Pro Tip**: Adjust stop loss higher as the stock moves in your favor!"""

async def run_chatbot():
    """Run interactive chatbot session"""
    bot = TradingChatbot()
    
    print("🤖 AuraTrade Trading Assistant")
    print("=" * 50)
    print("Hi! I'm your AI trading assistant. Ask me about:")
    print("• Stock analysis and recommendations")
    print("• Market prices and trends") 
    print("• Risk management and stop losses")
    print("• Growth and dividend stock ideas")
    print("\nType 'quit' to exit\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\n🤖 Bot: Thanks for using AuraTrade! Happy trading! 📈")
                break
            
            if not user_input:
                continue
            
            print("\n🤖 Bot: ", end="")
            response = await bot.process_message(user_input)
            print(response)
            print("\n" + "-" * 50 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n🤖 Bot: Goodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again.\n")

if __name__ == "__main__":
    asyncio.run(run_chatbot())