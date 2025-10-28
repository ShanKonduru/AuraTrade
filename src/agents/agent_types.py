from enum import Enum
from typing import Any, Dict, List, Optional


class AgentType(Enum):
    """Types of agents in the AuraTrade system"""
    DATA_INGESTION = "data_ingestion"
    TECHNICAL_ANALYSIS = "technical_analysis" 
    FUNDAMENTAL_ANALYSIS = "fundamental_analysis"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    ORCHESTRATOR = "orchestrator"
    EXECUTION = "execution"
    RISK_MANAGEMENT = "risk_management"


class MessageType(Enum):
    """Types of messages exchanged between agents"""
    DATA_REQUEST = "data_request"
    DATA_RESPONSE = "data_response"
    ANALYSIS_REQUEST = "analysis_request"
    ANALYSIS_RESPONSE = "analysis_response"
    TRADE_SIGNAL = "trade_signal"
    TRADE_EXECUTION = "trade_execution"
    RISK_CHECK = "risk_check"
    STATUS_UPDATE = "status_update"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


class ActionType(Enum):
    """Types of trading actions"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    ACCUMULATE = "ACCUMULATE"
    DIVEST = "DIVEST"


class SignalConfidence(Enum):
    """Signal confidence levels"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class MarketRegime(Enum):
    """Market regime classification"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    CRISIS = "crisis"


class RiskLevel(Enum):
    """Risk level classification"""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"