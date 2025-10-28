import os
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
try:
    from pydantic import BaseSettings, Field
except ImportError:
    # Fallback for environments without pydantic
    class BaseSettings:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    def Field(default=None, **kwargs):
        return default

try:
    from loguru import logger
except ImportError:
    # Fallback to standard logging
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

# Import all agent components
from src.agents.message_bus import MessageBus
from src.agents.perception.data_ingestion_agent import DataIngestionAgent
from src.agents.cognition.technical_analysis_agent import TechnicalAnalysisAgent
from src.agents.cognition.fundamental_analysis_agent import FundamentalAnalysisAgent
from src.agents.cognition.sentiment_analysis_agent import SentimentAnalysisAgent
from src.agents.decision.orchestrator_agent import OrchestratorAgent
from src.agents.action.execution_agent import ExecutionAgent
from src.risk.risk_management_agent import RiskManagementAgent
from src.agents.agent_types import AgentType, MessageType


class AuraTradeConfig(BaseSettings):
    """Configuration settings for AuraTrade platform"""
    
    # Environment
    environment: str = Field(default="development", env="AURA_ENVIRONMENT")
    log_level: str = Field(default="INFO", env="AURA_LOG_LEVEL")
    
    # API Keys
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    alpaca_api_key: Optional[str] = Field(default=None, env="ALPACA_API_KEY")
    alpaca_secret_key: Optional[str] = Field(default=None, env="ALPACA_SECRET_KEY")
    alpaca_paper: bool = Field(default=True, env="ALPACA_PAPER")
    
    # Trading Configuration
    max_position_size: float = Field(default=0.25, description="Maximum position size as % of portfolio")
    max_drawdown_limit: float = Field(default=0.10, description="Maximum portfolio drawdown")
    daily_loss_limit: float = Field(default=0.05, description="Maximum daily loss")
    risk_per_trade: float = Field(default=0.02, description="Risk per trade as % of portfolio")
    min_confidence_threshold: float = Field(default=0.3, description="Minimum confidence for trading")
    
    # Agent Weights
    technical_weight: float = Field(default=0.35, description="Weight for technical analysis")
    fundamental_weight: float = Field(default=0.40, description="Weight for fundamental analysis")
    sentiment_weight: float = Field(default=0.25, description="Weight for sentiment analysis")
    
    # Data Configuration
    cache_ttl: int = Field(default=300, description="Data cache TTL in seconds")
    news_hours_back: int = Field(default=24, description="Hours of news to analyze")
    
    # Broker Configuration
    broker_type: str = Field(default="paper_trading", description="Broker type: alpaca, paper_trading")
    initial_cash: float = Field(default=100000.0, description="Initial cash for paper trading")
    
    class Config:
        env_prefix = "AURA_"
        case_sensitive = False


@dataclass
class AgentStatus:
    """Status of an individual agent"""
    agent_type: str
    status: str
    last_heartbeat: datetime
    messages_processed: int
    errors_count: int


@dataclass
class SystemStatus:
    """Overall system status"""
    status: str  # "HEALTHY", "DEGRADED", "CRITICAL"
    agents: Dict[str, AgentStatus]
    trading_enabled: bool
    total_messages: int
    uptime_seconds: float
    last_updated: datetime


class AuraTradePlatform:
    """Main AuraTrade platform orchestrator"""
    
    def __init__(self, config: Optional[AuraTradeConfig] = None):
        self.config = config or AuraTradeConfig()
        self.message_bus = MessageBus()
        self.agents = {}
        self.system_status = None
        self.start_time = datetime.now()
        self.running = False
        
        # Initialize logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration"""
        log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        
        logger.remove()  # Remove default handler
        logger.add(
            sink="logs/auratrade_{time}.log",
            format=log_format,
            level=self.config.log_level,
            rotation="1 day",
            retention="30 days",
            compression="zip"
        )
        logger.add(
            sink=lambda msg: print(msg, end=""),
            format=log_format,
            level=self.config.log_level
        )
        
    async def initialize(self):
        """Initialize the AuraTrade platform"""
        logger.info("Initializing AuraTrade Platform...")
        
        try:
            # Start message bus
            await self.message_bus.start()
            
            # Initialize all agents
            await self._initialize_agents()
            
            # Start all agents
            await self._start_agents()
            
            logger.info("AuraTrade Platform initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize AuraTrade Platform: {e}")
            raise
            
    async def _initialize_agents(self):
        """Initialize all agents with their configurations"""
        
        # Agent configurations
        agent_configs = {
            'data_ingestion': {
                'cache_ttl': self.config.cache_ttl
            },
            'technical_analysis': {
                'rsi_period': 14,
                'ma_periods': [5, 10, 20, 50, 200],
                'bollinger_period': 20
            },
            'fundamental_analysis': {
                'discount_rate': 0.10,
                'terminal_growth_rate': 0.02,
                'openai_api_key': self.config.openai_api_key
            },
            'sentiment_analysis': {
                'news_hours_back': self.config.news_hours_back,
                'sentiment_threshold': 0.3
            },
            'orchestrator': {
                'openai_api_key': self.config.openai_api_key,
                'analysis_timeout': 30.0,
                'min_confidence_threshold': self.config.min_confidence_threshold,
                'agent_weights': {
                    AgentType.TECHNICAL_ANALYSIS: self.config.technical_weight,
                    AgentType.FUNDAMENTAL_ANALYSIS: self.config.fundamental_weight,
                    AgentType.SENTIMENT_ANALYSIS: self.config.sentiment_weight
                }
            },
            'execution': {
                'broker_type': self.config.broker_type,
                'alpaca_api_key': self.config.alpaca_api_key,
                'alpaca_secret_key': self.config.alpaca_secret_key,
                'alpaca_paper': self.config.alpaca_paper,
                'initial_cash': self.config.initial_cash,
                'max_position_size': self.config.max_position_size
            },
            'risk_management': {
                'max_position_size': self.config.max_position_size,
                'max_drawdown_limit': self.config.max_drawdown_limit,
                'daily_loss_limit': self.config.daily_loss_limit,
                'risk_per_trade': self.config.risk_per_trade
            }
        }
        
        # Create agents
        self.agents = {
            'data_ingestion': DataIngestionAgent(
                self.message_bus, 
                agent_configs['data_ingestion']
            ),
            'technical_analysis': TechnicalAnalysisAgent(
                self.message_bus, 
                agent_configs['technical_analysis']
            ),
            'fundamental_analysis': FundamentalAnalysisAgent(
                self.message_bus, 
                agent_configs['fundamental_analysis']
            ),
            'sentiment_analysis': SentimentAnalysisAgent(
                self.message_bus, 
                agent_configs['sentiment_analysis']
            ),
            'orchestrator': OrchestratorAgent(
                self.message_bus, 
                agent_configs['orchestrator']
            ),
            'execution': ExecutionAgent(
                self.message_bus, 
                agent_configs['execution']
            ),
            'risk_management': RiskManagementAgent(
                self.message_bus, 
                agent_configs['risk_management']
            )
        }
        
        logger.info(f"Initialized {len(self.agents)} agents")
        
    async def _start_agents(self):
        """Start all agents"""
        for agent_name, agent in self.agents.items():
            try:
                await agent.start()
                logger.info(f"Started {agent_name} agent")
            except Exception as e:
                logger.error(f"Failed to start {agent_name} agent: {e}")
                raise
                
    async def start(self):
        """Start the AuraTrade platform"""
        if self.running:
            logger.warning("Platform is already running")
            return
            
        logger.info("Starting AuraTrade Platform...")
        
        try:
            await self.initialize()
            self.running = True
            
            # Start monitoring task
            asyncio.create_task(self._monitor_system())
            
            logger.info("AuraTrade Platform started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start AuraTrade Platform: {e}")
            await self.stop()
            raise
            
    async def stop(self):
        """Stop the AuraTrade platform"""
        if not self.running:
            logger.warning("Platform is not running")
            return
            
        logger.info("Stopping AuraTrade Platform...")
        
        self.running = False
        
        # Stop all agents
        for agent_name, agent in self.agents.items():
            try:
                await agent.stop()
                logger.info(f"Stopped {agent_name} agent")
            except Exception as e:
                logger.error(f"Error stopping {agent_name} agent: {e}")
                
        # Stop message bus
        await self.message_bus.stop()
        
        logger.info("AuraTrade Platform stopped")
        
    async def _monitor_system(self):
        """Monitor system health and update status"""
        while self.running:
            try:
                await self._update_system_status()
                await asyncio.sleep(30)  # Update every 30 seconds
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                await asyncio.sleep(30)
                
    async def _update_system_status(self):
        """Update system status"""
        agent_statuses = {}
        healthy_agents = 0
        total_messages = 0
        
        for agent_name, agent in self.agents.items():
            status = agent.get_status()
            
            agent_status = AgentStatus(
                agent_type=status['agent_type'],
                status='HEALTHY' if status['is_running'] else 'STOPPED',
                last_heartbeat=datetime.fromisoformat(status['last_heartbeat']),
                messages_processed=status['messages_processed'],
                errors_count=status['errors_count']
            )
            
            agent_statuses[agent_name] = agent_status
            total_messages += status['messages_processed']
            
            if status['is_running'] and status['errors_count'] < 10:
                healthy_agents += 1
                
        # Determine overall system status
        if healthy_agents == len(self.agents):
            system_status = "HEALTHY"
        elif healthy_agents >= len(self.agents) * 0.7:
            system_status = "DEGRADED"
        else:
            system_status = "CRITICAL"
            
        # Check if trading should be enabled
        critical_agents = ['orchestrator', 'execution', 'risk_management']
        trading_enabled = all(
            agent_statuses.get(agent, AgentStatus('', 'STOPPED', datetime.now(), 0, 0)).status == 'HEALTHY'
            for agent in critical_agents
        )
        
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        self.system_status = SystemStatus(
            status=system_status,
            agents=agent_statuses,
            trading_enabled=trading_enabled,
            total_messages=total_messages,
            uptime_seconds=uptime,
            last_updated=datetime.now()
        )
        
    async def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        if self.system_status:
            return asdict(self.system_status)
        else:
            return {
                'status': 'OFFLINE',
                'agents': {},
                'trading_enabled': False,
                'total_messages': 0,
                'uptime_seconds': 0,
                'last_updated': datetime.now().isoformat()
            }
            
    async def request_trading_signal(self, symbol: str) -> Dict[str, Any]:
        """Request a trading signal for a symbol"""
        if not self.running or not self.system_status.trading_enabled:
            return {
                'success': False,
                'error': 'Trading is not enabled',
                'symbol': symbol
            }
            
        try:
            # Send trade signal request to orchestrator
            orchestrator = self.agents.get('orchestrator')
            if not orchestrator:
                return {
                    'success': False,
                    'error': 'Orchestrator agent not available',
                    'symbol': symbol
                }
                
            # Request trading signal
            response = await orchestrator.request_response(
                recipient=AgentType.ORCHESTRATOR,
                message_type=MessageType.TRADE_SIGNAL,
                payload={'symbol': symbol},
                timeout=60.0
            )
            
            if response and response.success:
                return {
                    'success': True,
                    'trading_decision': response.data,
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': response.error if response else 'No response from orchestrator',
                    'symbol': symbol
                }
                
        except Exception as e:
            logger.error(f"Error requesting trading signal for {symbol}: {e}")
            return {
                'success': False,
                'error': str(e),
                'symbol': symbol
            }
            
    async def execute_trade(self, trading_decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a trading decision"""
        if not self.running or not self.system_status.trading_enabled:
            return {
                'success': False,
                'error': 'Trading is not enabled'
            }
            
        try:
            # Send to risk management for pre-trade check
            risk_agent = self.agents.get('risk_management')
            if risk_agent:
                risk_response = await risk_agent.request_response(
                    recipient=AgentType.RISK_MANAGEMENT,
                    message_type=MessageType.RISK_CHECK,
                    payload={
                        'check_type': 'pre_trade',
                        'trading_decision': trading_decision
                    },
                    timeout=30.0
                )
                
                if not risk_response or not risk_response.success:
                    return {
                        'success': False,
                        'error': 'Risk check failed',
                        'details': risk_response.error if risk_response else 'No risk response'
                    }
                    
                if not risk_response.data.get('approved', False):
                    return {
                        'success': False,
                        'error': 'Trade rejected by risk management',
                        'risk_checks': risk_response.data.get('risk_checks', []),
                        'recommended_adjustments': risk_response.data.get('recommended_adjustments', [])
                    }
                    
            # Send to execution agent
            execution_agent = self.agents.get('execution')
            if not execution_agent:
                return {
                    'success': False,
                    'error': 'Execution agent not available'
                }
                
            execution_response = await execution_agent.request_response(
                recipient=AgentType.EXECUTION,
                message_type=MessageType.TRADE_EXECUTION,
                payload=trading_decision,
                timeout=30.0
            )
            
            if execution_response and execution_response.success:
                return {
                    'success': True,
                    'execution_result': execution_response.data,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': execution_response.error if execution_response else 'No execution response'
                }
                
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def run_trading_cycle(self, symbols: List[str]) -> Dict[str, Any]:
        """Run a complete trading cycle for given symbols"""
        if not symbols:
            return {'success': False, 'error': 'No symbols provided'}
            
        results = {}
        
        for symbol in symbols:
            try:
                logger.info(f"Running trading cycle for {symbol}")
                
                # Request trading signal
                signal_result = await self.request_trading_signal(symbol)
                
                if signal_result['success']:
                    trading_decision = signal_result['trading_decision']
                    
                    # Only execute if action is not HOLD
                    action = trading_decision.get('action', 'HOLD')
                    if action != 'HOLD':
                        execution_result = await self.execute_trade(trading_decision)
                        results[symbol] = {
                            'signal': signal_result,
                            'execution': execution_result
                        }
                    else:
                        results[symbol] = {
                            'signal': signal_result,
                            'execution': {'success': True, 'message': 'No action required (HOLD)'}
                        }
                else:
                    results[symbol] = {
                        'signal': signal_result,
                        'execution': {'success': False, 'error': 'No valid signal'}
                    }
                    
            except Exception as e:
                logger.error(f"Error in trading cycle for {symbol}: {e}")
                results[symbol] = {
                    'signal': {'success': False, 'error': str(e)},
                    'execution': {'success': False, 'error': 'Trading cycle failed'}
                }
                
        return {
            'success': True,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }


# Global platform instance
_platform_instance = None


def get_platform() -> AuraTradePlatform:
    """Get the global platform instance"""
    global _platform_instance
    if _platform_instance is None:
        _platform_instance = AuraTradePlatform()
    return _platform_instance


async def main():
    """Main entry point for AuraTrade platform"""
    # Load configuration
    config = AuraTradeConfig()
    
    # Create platform instance
    platform = AuraTradePlatform(config)
    
    try:
        # Start platform
        await platform.start()
        
        # Example trading cycle
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
        
        logger.info("Starting example trading cycle...")
        results = await platform.run_trading_cycle(symbols)
        
        logger.info("Trading cycle results:")
        for symbol, result in results['results'].items():
            logger.info(f"{symbol}: {result}")
            
        # Keep running
        logger.info("Platform running... Press Ctrl+C to stop")
        while True:
            await asyncio.sleep(60)  # Run trading cycle every minute
            
            # Check system status
            status = await platform.get_system_status()
            if status['status'] != 'HEALTHY':
                logger.warning(f"System status: {status['status']}")
                
            # Run trading cycle
            if status['trading_enabled']:
                await platform.run_trading_cycle(symbols)
            else:
                logger.warning("Trading is disabled")
                
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Platform error: {e}")
    finally:
        await platform.stop()


if __name__ == "__main__":
    asyncio.run(main())