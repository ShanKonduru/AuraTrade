"""
Configuration file for AuraTrade Platform
"""
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class TradingConfig:
    """Trading configuration parameters"""
    max_position_size: float = 0.25  # Maximum position size as % of portfolio
    max_drawdown_limit: float = 0.10  # Maximum portfolio drawdown
    daily_loss_limit: float = 0.05  # Maximum daily loss
    risk_per_trade: float = 0.02  # Risk per trade as % of portfolio
    min_confidence_threshold: float = 0.3  # Minimum confidence for trading


@dataclass
class AgentConfig:
    """Configuration for individual agents"""
    technical_weight: float = 0.35  # Weight for technical analysis
    fundamental_weight: float = 0.40  # Weight for fundamental analysis
    sentiment_weight: float = 0.25  # Weight for sentiment analysis


@dataclass
class DataConfig:
    """Data configuration parameters"""
    cache_ttl: int = 300  # Data cache TTL in seconds
    news_hours_back: int = 24  # Hours of news to analyze


@dataclass
class BrokerConfig:
    """Broker configuration"""
    broker_type: str = "paper_trading"  # Broker type: alpaca, paper_trading
    initial_cash: float = 100000.0  # Initial cash for paper trading
    alpaca_paper: bool = True  # Use Alpaca paper trading


@dataclass
class AuraTradeConfig:
    """Main configuration class for AuraTrade platform"""
    
    # Environment
    environment: str = "development"
    log_level: str = "INFO"
    
    # API Keys (loaded from environment variables)
    openai_api_key: Optional[str] = None
    alpaca_api_key: Optional[str] = None
    alpaca_secret_key: Optional[str] = None
    
    # Configuration sections
    trading: TradingConfig = TradingConfig()
    agents: AgentConfig = AgentConfig()
    data: DataConfig = DataConfig()
    broker: BrokerConfig = BrokerConfig()
    
    def __post_init__(self):
        """Load configuration from environment variables"""
        # Environment settings
        self.environment = os.getenv("AURA_ENVIRONMENT", self.environment)
        self.log_level = os.getenv("AURA_LOG_LEVEL", self.log_level)
        
        # API Keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.alpaca_api_key = os.getenv("ALPACA_API_KEY")
        self.alpaca_secret_key = os.getenv("ALPACA_SECRET_KEY")
        
        # Trading configuration
        self.trading.max_position_size = float(os.getenv("AURA_MAX_POSITION_SIZE", self.trading.max_position_size))
        self.trading.max_drawdown_limit = float(os.getenv("AURA_MAX_DRAWDOWN_LIMIT", self.trading.max_drawdown_limit))
        self.trading.daily_loss_limit = float(os.getenv("AURA_DAILY_LOSS_LIMIT", self.trading.daily_loss_limit))
        self.trading.risk_per_trade = float(os.getenv("AURA_RISK_PER_TRADE", self.trading.risk_per_trade))
        self.trading.min_confidence_threshold = float(os.getenv("AURA_MIN_CONFIDENCE", self.trading.min_confidence_threshold))
        
        # Agent weights
        self.agents.technical_weight = float(os.getenv("AURA_TECHNICAL_WEIGHT", self.agents.technical_weight))
        self.agents.fundamental_weight = float(os.getenv("AURA_FUNDAMENTAL_WEIGHT", self.agents.fundamental_weight))
        self.agents.sentiment_weight = float(os.getenv("AURA_SENTIMENT_WEIGHT", self.agents.sentiment_weight))
        
        # Data configuration
        self.data.cache_ttl = int(os.getenv("AURA_CACHE_TTL", self.data.cache_ttl))
        self.data.news_hours_back = int(os.getenv("AURA_NEWS_HOURS_BACK", self.data.news_hours_back))
        
        # Broker configuration
        self.broker.broker_type = os.getenv("AURA_BROKER_TYPE", self.broker.broker_type)
        self.broker.initial_cash = float(os.getenv("AURA_INITIAL_CASH", self.broker.initial_cash))
        self.broker.alpaca_paper = os.getenv("ALPACA_PAPER", "true").lower() == "true"
        
    def get_agent_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get configuration dictionaries for each agent"""
        return {
            'data_ingestion': {
                'cache_ttl': self.data.cache_ttl
            },
            'technical_analysis': {
                'rsi_period': 14,
                'ma_periods': [5, 10, 20, 50, 200],
                'bollinger_period': 20
            },
            'fundamental_analysis': {
                'discount_rate': 0.10,
                'terminal_growth_rate': 0.02,
                'openai_api_key': self.openai_api_key
            },
            'sentiment_analysis': {
                'news_hours_back': self.data.news_hours_back,
                'sentiment_threshold': 0.3
            },
            'orchestrator': {
                'openai_api_key': self.openai_api_key,
                'analysis_timeout': 30.0,
                'min_confidence_threshold': self.trading.min_confidence_threshold,
                'agent_weights': {
                    'technical_analysis': self.agents.technical_weight,
                    'fundamental_analysis': self.agents.fundamental_weight,
                    'sentiment_analysis': self.agents.sentiment_weight
                }
            },
            'execution': {
                'broker_type': self.broker.broker_type,
                'alpaca_api_key': self.alpaca_api_key,
                'alpaca_secret_key': self.alpaca_secret_key,
                'alpaca_paper': self.broker.alpaca_paper,
                'initial_cash': self.broker.initial_cash,
                'max_position_size': self.trading.max_position_size
            },
            'risk_management': {
                'max_position_size': self.trading.max_position_size,
                'max_drawdown_limit': self.trading.max_drawdown_limit,
                'daily_loss_limit': self.trading.daily_loss_limit,
                'risk_per_trade': self.trading.risk_per_trade
            }
        }
        
    def validate(self) -> bool:
        """Validate configuration"""
        # Check required API keys for production
        if self.environment == "production":
            if not self.openai_api_key:
                raise ValueError("OpenAI API key is required for production")
            if not self.alpaca_api_key or not self.alpaca_secret_key:
                raise ValueError("Alpaca API keys are required for production")
                
        # Validate weights sum to 1.0
        total_weight = self.agents.technical_weight + self.agents.fundamental_weight + self.agents.sentiment_weight
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Agent weights must sum to 1.0, got {total_weight}")
            
        # Validate trading parameters
        if not 0 < self.trading.max_position_size <= 1.0:
            raise ValueError("max_position_size must be between 0 and 1")
        if not 0 < self.trading.max_drawdown_limit <= 1.0:
            raise ValueError("max_drawdown_limit must be between 0 and 1")
        if not 0 < self.trading.daily_loss_limit <= 1.0:
            raise ValueError("daily_loss_limit must be between 0 and 1")
            
        return True
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'environment': self.environment,
            'log_level': self.log_level,
            'has_openai_key': bool(self.openai_api_key),
            'has_alpaca_keys': bool(self.alpaca_api_key and self.alpaca_secret_key),
            'trading': {
                'max_position_size': self.trading.max_position_size,
                'max_drawdown_limit': self.trading.max_drawdown_limit,
                'daily_loss_limit': self.trading.daily_loss_limit,
                'risk_per_trade': self.trading.risk_per_trade,
                'min_confidence_threshold': self.trading.min_confidence_threshold
            },
            'agents': {
                'technical_weight': self.agents.technical_weight,
                'fundamental_weight': self.agents.fundamental_weight,
                'sentiment_weight': self.agents.sentiment_weight
            },
            'data': {
                'cache_ttl': self.data.cache_ttl,
                'news_hours_back': self.data.news_hours_back
            },
            'broker': {
                'broker_type': self.broker.broker_type,
                'initial_cash': self.broker.initial_cash,
                'alpaca_paper': self.broker.alpaca_paper
            }
        }


def load_config() -> AuraTradeConfig:
    """Load configuration from environment and return AuraTradeConfig instance"""
    config = AuraTradeConfig()
    config.validate()
    return config