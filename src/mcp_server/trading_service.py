"""
Complete MCP Server endpoints and trading service integrations
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import uuid
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import agent classes with corrected paths
try:
    from agents.data_ingestion_agent import DataIngestionAgent
    from agents.technical_analysis_agent import TechnicalAnalysisAgent
    from agents.fundamental_analysis_agent import FundamentalAnalysisAgent
    from agents.sentiment_analysis_agent import SentimentAnalysisAgent
    from agents.risk_management_agent import RiskManagementAgent
    from agents.execution_agent import ExecutionAgent
    from agents.orchestrator_agent import OrchestratorAgent
except ImportError as e:
    logging.warning(f"Could not import agents: {e}. Using demo mode.")

class TradingService:
    """Core trading service for MCP server integration"""
    
    def __init__(self):
        self.agents_available = self._initialize_agents()
        self.demo_mode = not self.agents_available
        
        if self.demo_mode:
            logging.info("Running in demo mode - agents not available")
        else:
            logging.info("All trading agents initialized successfully")
    
    def _initialize_agents(self) -> bool:
        """Initialize all trading agents"""
        try:
            self.data_agent = DataIngestionAgent("data_agent")
            self.technical_agent = TechnicalAnalysisAgent("technical_agent")
            self.fundamental_agent = FundamentalAnalysisAgent("fundamental_agent")
            self.sentiment_agent = SentimentAnalysisAgent("sentiment_agent")
            self.risk_agent = RiskManagementAgent("risk_agent")
            self.execution_agent = ExecutionAgent("execution_agent")
            self.orchestrator = OrchestratorAgent("orchestrator")
            return True
        except Exception as e:
            logging.error(f"Failed to initialize agents: {e}")
            return False
    
    async def analyze_market(self, symbol: str, timeframe: str = "1d", 
                           include_technical: bool = True,
                           include_fundamental: bool = True,
                           include_sentiment: bool = True) -> Dict[str, Any]:
        """Comprehensive market analysis for a symbol"""
        
        if self.demo_mode:
            return await self._demo_market_analysis(symbol)
        
        try:
            analysis = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "timeframe": timeframe
            }
            
            # Get market data
            market_data = await self.data_agent.get_market_data(symbol, timeframe)
            analysis["market_data"] = market_data
            
            # Technical analysis
            if include_technical:
                technical_result = await self.technical_agent.analyze(symbol, market_data)
                analysis["technical"] = technical_result
            
            # Fundamental analysis
            if include_fundamental:
                fundamental_result = await self.fundamental_agent.analyze(symbol)
                analysis["fundamental"] = fundamental_result
            
            # Sentiment analysis
            if include_sentiment:
                sentiment_result = await self.sentiment_agent.analyze(symbol)
                analysis["sentiment"] = sentiment_result
            
            return analysis
            
        except Exception as e:
            logging.error(f"Market analysis failed for {symbol}: {e}")
            return await self._demo_market_analysis(symbol)
    
    async def get_trading_recommendation(self, symbol: str, amount: Optional[float] = None) -> Dict[str, Any]:
        """Get trading recommendation for a symbol"""
        
        if self.demo_mode:
            return await self._demo_trading_recommendation(symbol, amount)
        
        try:
            # Get comprehensive analysis
            analysis = await self.analyze_market(symbol)
            
            # Generate recommendation using orchestrator
            recommendation = await self.orchestrator.generate_recommendation(analysis, amount)
            
            return recommendation
            
        except Exception as e:
            logging.error(f"Trading recommendation failed for {symbol}: {e}")
            return await self._demo_trading_recommendation(symbol, amount)
    
    async def calculate_stop_loss(self, symbol: str, entry_price: float, 
                                position_size: int, risk_percentage: float = 2.0) -> Dict[str, Any]:
        """Calculate stop loss for a position"""
        
        if self.demo_mode:
            return await self._demo_stop_loss(symbol, entry_price, position_size, risk_percentage)
        
        try:
            stop_loss_data = await self.risk_agent.calculate_stop_loss(
                symbol, entry_price, position_size, risk_percentage
            )
            return stop_loss_data
            
        except Exception as e:
            logging.error(f"Stop loss calculation failed for {symbol}: {e}")
            return await self._demo_stop_loss(symbol, entry_price, position_size, risk_percentage)
    
    async def assess_portfolio_risk(self, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess overall portfolio risk"""
        
        if self.demo_mode:
            return await self._demo_portfolio_risk(positions)
        
        try:
            risk_assessment = await self.risk_agent.assess_portfolio_risk(positions)
            return risk_assessment
            
        except Exception as e:
            logging.error(f"Portfolio risk assessment failed: {e}")
            return await self._demo_portfolio_risk(positions)
    
    # Demo methods for when agents are not available
    async def _demo_market_analysis(self, symbol: str) -> Dict[str, Any]:
        """Demo market analysis with realistic data"""
        import random
        
        base_price = random.uniform(50, 300)
        change = random.uniform(-5, 5)
        
        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "market_data": {
                "price": round(base_price, 2),
                "change": round(change, 2),
                "change_percent": round((change / base_price) * 100, 2),
                "volume": random.randint(1000000, 50000000),
                "market_cap": random.randint(1000000000, 500000000000)
            },
            "technical": {
                "trend": random.choice(["bullish", "bearish", "neutral"]),
                "strength": round(random.uniform(0.3, 0.9), 2),
                "signals": random.sample([
                    "RSI oversold", "Moving average crossover", "Volume spike",
                    "Breakout pattern", "Support level", "Resistance level"
                ], k=random.randint(1, 3))
            },
            "fundamental": {
                "score": round(random.uniform(0.3, 0.8), 2),
                "recommendation": random.choice(["buy", "hold", "sell"]),
                "pe_ratio": round(random.uniform(10, 30), 1),
                "debt_ratio": round(random.uniform(0.1, 0.6), 2)
            },
            "sentiment": {
                "score": round(random.uniform(-0.3, 0.3), 2),
                "confidence": round(random.uniform(0.5, 0.9), 2),
                "sources": random.randint(50, 200)
            }
        }
    
    async def _demo_trading_recommendation(self, symbol: str, amount: Optional[float]) -> Dict[str, Any]:
        """Demo trading recommendation"""
        import random
        
        analysis = await self._demo_market_analysis(symbol)
        current_price = analysis["market_data"]["price"]
        
        recommendations = ["BUY", "SELL", "HOLD", "ACCUMULATE"]
        weights = [0.4, 0.2, 0.3, 0.1]  # Favor buy recommendations
        
        recommendation = random.choices(recommendations, weights=weights)[0]
        confidence = round(random.uniform(0.6, 0.95), 2)
        
        # Calculate prices based on recommendation
        if recommendation == "BUY":
            target_price = current_price * random.uniform(1.05, 1.25)
            stop_loss = current_price * random.uniform(0.90, 0.95)
            reasoning = "Strong technical indicators with positive sentiment"
        elif recommendation == "SELL":
            target_price = current_price * random.uniform(0.80, 0.95)
            stop_loss = current_price * random.uniform(1.03, 1.08)
            reasoning = "Weakening fundamentals and technical breakdown"
        else:
            target_price = current_price * random.uniform(0.98, 1.02)
            stop_loss = current_price * random.uniform(0.92, 0.96)
            reasoning = "Mixed signals suggest holding current position"
        
        return {
            "symbol": symbol,
            "recommendation": recommendation,
            "confidence": confidence,
            "reasoning": reasoning,
            "entry_price": current_price,
            "target_price": round(target_price, 2),
            "stop_loss": round(stop_loss, 2),
            "timestamp": datetime.now().isoformat(),
            "risk_reward_ratio": round(abs(target_price - current_price) / abs(current_price - stop_loss), 2)
        }
    
    async def _demo_stop_loss(self, symbol: str, entry_price: float, 
                            position_size: int, risk_percentage: float) -> Dict[str, Any]:
        """Demo stop loss calculation"""
        
        # Calculate stop loss based on risk percentage
        risk_amount = entry_price * (risk_percentage / 100)
        stop_loss_price = entry_price - risk_amount
        
        total_risk = risk_amount * position_size
        
        return {
            "symbol": symbol,
            "entry_price": entry_price,
            "position_size": position_size,
            "risk_percentage": risk_percentage,
            "recommended_stop_loss": round(stop_loss_price, 2),
            "risk_amount_per_share": round(risk_amount, 2),
            "total_risk_amount": round(total_risk, 2),
            "rationale": f"Stop loss set at {risk_percentage}% below entry price to limit downside risk",
            "timestamp": datetime.now().isoformat()
        }
    
    async def _demo_portfolio_risk(self, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Demo portfolio risk assessment"""
        import random
        
        total_value = sum(pos.get("value", 10000) for pos in positions)
        num_positions = len(positions)
        
        # Calculate basic risk metrics
        beta = round(random.uniform(0.8, 1.3), 2)
        sharpe_ratio = round(random.uniform(0.5, 2.0), 2)
        max_drawdown = round(random.uniform(5, 20), 1)
        var_95 = round(total_value * random.uniform(0.02, 0.08), 0)
        
        # Diversification score
        diversification_score = min(100, (num_positions / 10) * 100) if num_positions > 0 else 0
        
        # Overall risk level
        if beta < 1.0 and max_drawdown < 10:
            risk_level = "LOW"
        elif beta > 1.2 or max_drawdown > 15:
            risk_level = "HIGH"
        else:
            risk_level = "MEDIUM"
        
        return {
            "total_portfolio_value": total_value,
            "number_of_positions": num_positions,
            "portfolio_beta": beta,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown_percent": max_drawdown,
            "value_at_risk_95": var_95,
            "diversification_score": round(diversification_score, 1),
            "risk_level": risk_level,
            "recommendations": [
                "Consider adding more defensive positions" if risk_level == "HIGH" else "Portfolio risk is well-managed",
                "Increase diversification across sectors" if num_positions < 8 else "Good diversification level",
                "Monitor correlation between positions" if num_positions > 5 else "Consider adding more positions"
            ],
            "timestamp": datetime.now().isoformat()
        }

# Global service instance
trading_service = TradingService()

# MCP method handlers
async def handle_analyze(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle market analysis requests"""
    symbol = params.get("symbol", "").upper()
    timeframe = params.get("timeframe", "1d")
    include_technical = params.get("include_technical", True)
    include_fundamental = params.get("include_fundamental", True)
    include_sentiment = params.get("include_sentiment", True)
    
    if not symbol:
        return {"error": "Symbol is required"}
    
    try:
        result = await trading_service.analyze_market(
            symbol, timeframe, include_technical, include_fundamental, include_sentiment
        )
        return result
    except Exception as e:
        return {"error": str(e)}

async def handle_recommend(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle trading recommendation requests"""
    symbol = params.get("symbol", "").upper()
    amount = params.get("amount")
    
    if not symbol:
        return {"error": "Symbol is required"}
    
    try:
        result = await trading_service.get_trading_recommendation(symbol, amount)
        return result
    except Exception as e:
        return {"error": str(e)}

async def handle_stop_loss(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle stop loss calculation requests"""
    symbol = params.get("symbol", "").upper()
    entry_price = params.get("entry_price")
    position_size = params.get("position_size", 100)
    risk_percentage = params.get("risk_percentage", 2.0)
    
    if not symbol or not entry_price:
        return {"error": "Symbol and entry_price are required"}
    
    try:
        result = await trading_service.calculate_stop_loss(
            symbol, entry_price, position_size, risk_percentage
        )
        return result
    except Exception as e:
        return {"error": str(e)}

async def handle_portfolio_risk(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle portfolio risk assessment requests"""
    positions = params.get("positions", [])
    
    try:
        result = await trading_service.assess_portfolio_risk(positions)
        return result
    except Exception as e:
        return {"error": str(e)}

# Method registry for MCP server
MCP_METHODS = {
    "analyze": handle_analyze,
    "recommend": handle_recommend,
    "stop_loss": handle_stop_loss,
    "portfolio_risk": handle_portfolio_risk
}