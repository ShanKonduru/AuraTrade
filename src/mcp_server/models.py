"""
MCP Server Models and Types
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

class MCPMessageType(str, Enum):
    """MCP message types"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"

class TradingAction(str, Enum):
    """Trading actions"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    ACCUMULATE = "accumulate"
    DIVEST = "divest"

class RiskTolerance(str, Enum):
    """Risk tolerance levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class InvestmentHorizon(str, Enum):
    """Investment time horizons"""
    SHORT = "short"    # < 6 months
    MEDIUM = "medium"  # 6 months - 2 years
    LONG = "long"      # > 2 years

class MarketDataPoint(BaseModel):
    """Market data point"""
    symbol: str
    price: float
    change: float = 0.0
    change_percent: float = 0.0
    volume: int = 0
    timestamp: datetime = Field(default_factory=datetime.now)

class TechnicalIndicators(BaseModel):
    """Technical analysis indicators"""
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_lower: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    volume_sma: Optional[float] = None

class TechnicalAnalysis(BaseModel):
    """Technical analysis result"""
    indicators: TechnicalIndicators
    signals: List[str] = []
    trend: str = "neutral"
    strength: float = Field(ge=0.0, le=1.0, default=0.5)
    support_levels: List[float] = []
    resistance_levels: List[float] = []

class FundamentalRatios(BaseModel):
    """Fundamental analysis ratios"""
    pe_ratio: Optional[float] = None
    peg_ratio: Optional[float] = None
    price_to_book: Optional[float] = None
    price_to_sales: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None

class FundamentalAnalysis(BaseModel):
    """Fundamental analysis result"""
    ratios: FundamentalRatios
    valuation: Dict[str, Any] = {}
    score: float = Field(ge=0.0, le=1.0, default=0.5)
    recommendation: TradingAction = TradingAction.HOLD
    target_price: Optional[float] = None
    fair_value: Optional[float] = None

class SentimentAnalysis(BaseModel):
    """Sentiment analysis result"""
    score: float = Field(ge=-1.0, le=1.0, default=0.0)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    sources: List[str] = []
    summary: str = "Neutral sentiment"
    news_count: int = 0
    social_mentions: int = 0

class MarketAnalysisResult(BaseModel):
    """Complete market analysis result"""
    symbol: str
    timestamp: datetime = Field(default_factory=datetime.now)
    market_data: MarketDataPoint
    technical: Optional[TechnicalAnalysis] = None
    fundamental: Optional[FundamentalAnalysis] = None
    sentiment: Optional[SentimentAnalysis] = None
    overall_score: float = Field(ge=0.0, le=1.0, default=0.5)
    recommendation: TradingAction = TradingAction.HOLD
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)

class TradingRecommendation(BaseModel):
    """Trading recommendation"""
    symbol: str
    action: TradingAction
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    entry_price: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    position_size: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    max_risk: Optional[float] = None
    time_horizon: InvestmentHorizon = InvestmentHorizon.MEDIUM

class StopLossCalculation(BaseModel):
    """Stop loss calculation result"""
    symbol: str
    entry_price: float
    recommended_stop_loss: float
    risk_amount: float
    risk_percentage: float
    volatility_stop: float
    percentage_stop: float
    rationale: str

class PortfolioPosition(BaseModel):
    """Portfolio position"""
    symbol: str
    quantity: float
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    weight: float  # Portfolio weight percentage

class RiskMetrics(BaseModel):
    """Portfolio risk metrics"""
    beta: float = 1.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    var_95: float = 0.0  # Value at Risk (95% confidence)
    concentration_risk: float = 0.0
    volatility: float = 0.0

class PortfolioRiskAssessment(BaseModel):
    """Portfolio risk assessment result"""
    total_value: float
    positions: List[PortfolioPosition]
    risk_metrics: RiskMetrics
    diversification_score: float = Field(ge=0.0, le=1.0, default=0.5)
    sector_allocation: Dict[str, float] = {}
    recommendations: List[str] = []
    risk_level: str = "medium"

class MCPError(BaseModel):
    """MCP error response"""
    code: int
    message: str
    data: Optional[Dict[str, Any]] = None

class MCPRequest(BaseModel):
    """MCP request message"""
    jsonrpc: str = "2.0"
    id: Union[str, int]
    method: str
    params: Dict[str, Any] = {}

class MCPResponse(BaseModel):
    """MCP response message"""
    jsonrpc: str = "2.0"
    id: Union[str, int]
    result: Optional[Dict[str, Any]] = None
    error: Optional[MCPError] = None

class MCPNotification(BaseModel):
    """MCP notification message"""
    jsonrpc: str = "2.0"
    method: str
    params: Dict[str, Any] = {}

# Service request models
class MarketAnalysisRequest(BaseModel):
    """Market analysis request"""
    symbol: str = Field(..., description="Stock symbol to analyze")
    timeframe: str = Field(default="1d", description="Analysis timeframe")
    include_technical: bool = Field(default=True, description="Include technical analysis")
    include_fundamental: bool = Field(default=True, description="Include fundamental analysis")
    include_sentiment: bool = Field(default=True, description="Include sentiment analysis")

class TradingRecommendationRequest(BaseModel):
    """Trading recommendation request"""
    symbol: str = Field(..., description="Stock symbol")
    amount: Optional[float] = Field(None, description="Investment amount")
    risk_tolerance: RiskTolerance = Field(default=RiskTolerance.MEDIUM)
    investment_horizon: InvestmentHorizon = Field(default=InvestmentHorizon.MEDIUM)

class StopLossRequest(BaseModel):
    """Stop loss calculation request"""
    symbol: str = Field(..., description="Stock symbol")
    entry_price: float = Field(..., description="Entry price")
    position_size: float = Field(..., description="Position size (number of shares)")
    risk_percentage: float = Field(default=2.0, description="Risk percentage (default 2%)")

class PortfolioRiskRequest(BaseModel):
    """Portfolio risk assessment request"""
    positions: List[Dict[str, Any]] = Field(..., description="List of portfolio positions")
    total_value: float = Field(..., description="Total portfolio value")

class GrowthStocksRequest(BaseModel):
    """Growth stocks screening request"""
    max_price: Optional[float] = Field(None, description="Maximum stock price")
    min_revenue_growth: float = Field(default=15.0, description="Minimum revenue growth %")
    min_earnings_growth: float = Field(default=20.0, description="Minimum earnings growth %")
    market_cap_min: Optional[float] = Field(None, description="Minimum market cap")
    sector: Optional[str] = Field(None, description="Specific sector filter")
    limit: int = Field(default=10, description="Number of results to return")

class DividendStocksRequest(BaseModel):
    """Dividend stocks screening request"""
    min_yield: float = Field(default=3.0, description="Minimum dividend yield %")
    min_payout_ratio: float = Field(default=0.3, description="Minimum payout ratio")
    max_payout_ratio: float = Field(default=0.8, description="Maximum payout ratio")
    years_of_growth: int = Field(default=5, description="Years of consecutive dividend growth")
    limit: int = Field(default=10, description="Number of results to return")