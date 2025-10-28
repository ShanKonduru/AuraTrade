"""
AuraTrade MCP Server
Model Context Protocol server for trading intelligence services
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
from pathlib import Path

# Add src to path for AuraTrade imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_server.trading_service import MCP_METHODS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for MCP protocol
class MCPRequest(BaseModel):
    """MCP request format"""
    id: str
    method: str
    params: Dict[str, Any] = {}

class MCPResponse(BaseModel):
    """MCP response format"""
    id: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None

class MarketAnalysisRequest(BaseModel):
    """Market analysis request"""
    symbol: str
    timeframe: str = "1d"
    include_technical: bool = True
    include_fundamental: bool = True
    include_sentiment: bool = True

class TradingRecommendationRequest(BaseModel):
    """Trading recommendation request"""
    symbol: str
    amount: Optional[float] = None
    risk_tolerance: str = "medium"  # low, medium, high
    investment_horizon: str = "medium"  # short, medium, long

class StopLossRequest(BaseModel):
    """Stop loss calculation request"""
    symbol: str
    entry_price: float
    position_size: float
    risk_percentage: float = 2.0  # Default 2% risk

class PortfolioRiskRequest(BaseModel):
    """Portfolio risk assessment request"""
    positions: List[Dict[str, Any]]
    total_value: float

@dataclass  
class DemoTradingService:
    """Demo trading service for development and testing"""
    
    def __init__(self):
        logger.info("Initializing demo trading service")
    
    async def analyze_market(self, request: MarketAnalysisRequest) -> Dict[str, Any]:
        """Get comprehensive market analysis with demo data"""
        try:
            import random
            
            # Generate realistic demo data
            symbol = request.symbol
            base_price = random.uniform(50, 300)
            change = random.uniform(-5, 5)
            
            analysis = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "market_data": {
                    "price": round(base_price, 2),
                    "change": round(change, 2),
                    "change_percent": round((change / base_price) * 100, 2),
                    "volume": random.randint(1000000, 50000000),
                    "market_cap": random.randint(1000000000, 500000000000)
                }
            }
            
            # Technical analysis (demo)
            if request.include_technical:
                analysis["technical"] = {
                    "trend": random.choice(["bullish", "bearish", "neutral"]),
                    "strength": round(random.uniform(0.3, 0.9), 2),
                    "signals": random.sample([
                        "RSI oversold", "Moving average crossover", "Volume spike",
                        "Breakout pattern", "Support level", "Resistance level"
                    ], k=random.randint(1, 3))
                }
            
            # Fundamental analysis (demo)
            if request.include_fundamental:
                analysis["fundamental"] = {
                    "score": round(random.uniform(0.3, 0.8), 2),
                    "recommendation": random.choice(["buy", "hold", "sell"]),
                    "pe_ratio": round(random.uniform(10, 30), 1),
                    "debt_ratio": round(random.uniform(0.1, 0.6), 2)
                }
            
            # Sentiment analysis (demo)
            if request.include_sentiment:
                analysis["sentiment"] = {
                    "score": round(random.uniform(-0.3, 0.3), 2),
                    "confidence": round(random.uniform(0.5, 0.9), 2),
                    "sources": random.randint(50, 200)
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in market analysis: {e}")
            return {"error": str(e)}
    
    async def get_trading_recommendation(self, request: TradingRecommendationRequest) -> Dict[str, Any]:
        """Get trading recommendation with demo data"""
        try:
            import random
            
            symbol = request.symbol
            
            # Generate recommendation
            recommendations = ["BUY", "SELL", "HOLD", "ACCUMULATE"]
            weights = [0.4, 0.2, 0.3, 0.1]  # Favor buy recommendations
            
            recommendation = random.choices(recommendations, weights=weights)[0]
            confidence = round(random.uniform(0.6, 0.95), 2)
            
            # Demo pricing
            current_price = random.uniform(50, 300)
            
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
                "entry_price": round(current_price, 2),
                "target_price": round(target_price, 2),
                "stop_loss": round(stop_loss, 2),
                "timestamp": datetime.now().isoformat(),
                "risk_reward_ratio": round(abs(target_price - current_price) / abs(current_price - stop_loss), 2)
            }
            
        except Exception as e:
            logger.error(f"Error in trading recommendation: {e}")
            return {"error": str(e)}
    
    async def calculate_stop_loss(self, request: StopLossRequest) -> Dict[str, Any]:
        """Calculate optimal stop loss with demo data"""
        try:
            # Calculate stop loss based on risk percentage
            risk_amount = request.entry_price * (request.risk_percentage / 100)
            stop_loss_price = request.entry_price - risk_amount
            
            total_risk = risk_amount * request.position_size
            
            return {
                "symbol": request.symbol,
                "entry_price": request.entry_price,
                "position_size": request.position_size,
                "risk_percentage": request.risk_percentage,
                "recommended_stop_loss": round(stop_loss_price, 2),
                "risk_amount_per_share": round(risk_amount, 2),
                "total_risk_amount": round(total_risk, 2),
                "rationale": f"Stop loss set at {request.risk_percentage}% below entry price to limit downside risk",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in stop loss calculation: {e}")
            return {"error": str(e)}
    
    async def assess_portfolio_risk(self, request: PortfolioRiskRequest) -> Dict[str, Any]:
        """Assess portfolio risk with demo data"""
        try:
            import random
            
            total_value = request.total_value
            num_positions = len(request.positions)
            
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
            
        except Exception as e:
            logger.error(f"Error in portfolio risk assessment: {e}")
            return {"error": str(e)}


class AuraTradeMCPServer:
    """MCP Server for AuraTrade platform"""
    
    def __init__(self):
        """Initialize server in demo mode"""
        self.app = FastAPI(title="AuraTrade MCP Server", version="1.0.0")
        self.trading_service = DemoTradingService()
        self.setup_middleware()
        self.setup_routes()
    
    def setup_middleware(self):
        """Setup CORS and other middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.on_event("startup")
        async def startup():
            """Initialize demo trading service on startup"""
            logger.info("AuraTrade MCP Server started in demo mode")
        
        @self.app.on_event("shutdown")
        async def shutdown():
            """Cleanup on shutdown"""
            logger.info("AuraTrade MCP Server shutting down")
        
        @self.app.get("/")
        async def root():
            """Root endpoint"""
            return {
                "service": "AuraTrade MCP Server",
                "version": "1.0.0",
                "status": "running",
                "mode": "demo",
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "service": "AuraTrade MCP Server",
                "mode": "demo",
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.post("/mcp", response_model=MCPResponse)
        async def handle_mcp_request(request: MCPRequest):
            """Handle MCP protocol requests"""
            try:
                if request.method == "analyze":
                    analysis_req = MarketAnalysisRequest(**request.params)
                    result = await self.trading_service.analyze_market(analysis_req)
                    
                elif request.method == "recommend":
                    rec_req = TradingRecommendationRequest(**request.params)
                    result = await self.trading_service.get_trading_recommendation(rec_req)
                    
                elif request.method == "stop_loss":
                    stop_req = StopLossRequest(**request.params)
                    result = await self.trading_service.calculate_stop_loss(stop_req)
                    
                elif request.method == "risk_assess":
                    risk_req = PortfolioRiskRequest(**request.params)
                    result = await self.trading_service.assess_portfolio_risk(risk_req)
                    
                else:
                    result = {"error": f"Unknown method: {request.method}"}
                
                return MCPResponse(id=request.id, result=result)
                
            except Exception as e:
                logger.error(f"Error handling MCP request: {e}")
                return MCPResponse(
                    id=request.id,
                    error={"code": -1, "message": str(e)}
                )
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time data"""
            await websocket.accept()
            try:
                while True:
                    # Send periodic market updates
                    data = await websocket.receive_text()
                    request = json.loads(data)
                    
                    # Handle real-time requests
                    response = {"type": "market_update", "data": {}}
                    await websocket.send_text(json.dumps(response))
                    
            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected")
            except Exception as e:
                logger.error(f"WebSocket error: {e}")

def create_server() -> AuraTradeMCPServer:
    """Create MCP server instance"""
    return AuraTradeMCPServer()

def run_server(host: str = "localhost", port: int = 8000):
    """Run the MCP server"""
    server = create_server()
    
    logger.info(f"Starting AuraTrade MCP Server on {host}:{port}")
    uvicorn.run(
        server.app,
        host=host,
        port=port,
        log_level="info"
    )

# Create app instance for uvicorn
server = create_server()
app = server.app

if __name__ == "__main__":
    run_server()