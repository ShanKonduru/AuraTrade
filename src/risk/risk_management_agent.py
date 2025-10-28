import asyncio
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
from loguru import logger

from ..base_agent import BaseAgent, AgentMessage, AgentResponse
from ...agent_types import AgentType, MessageType, ActionType, RiskLevel


class RiskType(Enum):
    """Types of risk"""
    POSITION_SIZE = "position_size"
    CONCENTRATION = "concentration"
    DRAWDOWN = "drawdown"
    VOLATILITY = "volatility"
    LIQUIDITY = "liquidity"
    CORRELATION = "correlation"
    LEVERAGE = "leverage"
    MARKET_CONDITION = "market_condition"


class RiskStatus(Enum):
    """Risk status levels"""
    GREEN = "green"      # Normal operations
    YELLOW = "yellow"    # Caution required
    ORANGE = "orange"    # High risk, reduce exposure
    RED = "red"          # Critical risk, halt trading


@dataclass
class RiskLimit:
    """Risk limit definition"""
    name: str
    risk_type: RiskType
    limit_value: float
    current_value: float
    breach_threshold: float
    warning_threshold: float
    status: RiskStatus
    description: str


@dataclass
class RiskMetrics:
    """Portfolio risk metrics"""
    total_exposure: float
    max_drawdown: float
    current_drawdown: float
    portfolio_var: float  # Value at Risk
    portfolio_volatility: float
    sharpe_ratio: float
    beta: float
    largest_position_pct: float
    correlation_risk: float
    cash_ratio: float
    updated_at: datetime


@dataclass
class RiskAlert:
    """Risk alert structure"""
    alert_id: str
    risk_type: RiskType
    severity: RiskLevel
    message: str
    recommended_action: str
    affected_symbols: List[str]
    created_at: datetime
    acknowledged: bool = False


class PositionSizeCalculator:
    """Calculate position sizes based on risk parameters"""
    
    def __init__(self, config: Dict[str, Any]):
        self.max_position_size = config.get('max_position_size', 0.25)  # 25% max
        self.max_sector_exposure = config.get('max_sector_exposure', 0.40)  # 40% max
        self.risk_per_trade = config.get('risk_per_trade', 0.02)  # 2% risk per trade
        self.volatility_adjustment = config.get('volatility_adjustment', True)
        
    def calculate_position_size(self, 
                              portfolio_value: float,
                              confidence: float,
                              stop_loss_distance: float,
                              current_volatility: float = None) -> Dict[str, Any]:
        """Calculate optimal position size based on risk parameters"""
        
        # Base position size from confidence
        base_size = self.max_position_size * confidence
        
        # Risk-based position sizing (Kelly Criterion simplified)
        if stop_loss_distance > 0:
            risk_based_size = (self.risk_per_trade * portfolio_value) / stop_loss_distance
            risk_based_size_pct = risk_based_size / portfolio_value
        else:
            risk_based_size_pct = base_size
            
        # Volatility adjustment
        if self.volatility_adjustment and current_volatility:
            # Reduce position size for high volatility
            vol_adjustment = 1.0 / (1.0 + current_volatility)
            adjusted_size = base_size * vol_adjustment
        else:
            adjusted_size = base_size
            
        # Take the minimum of all constraints
        final_size = min(base_size, risk_based_size_pct, adjusted_size, self.max_position_size)
        
        return {
            'recommended_size_pct': final_size * 100,
            'base_size_pct': base_size * 100,
            'risk_based_size_pct': risk_based_size_pct * 100,
            'volatility_adjusted_pct': adjusted_size * 100,
            'max_allowed_pct': self.max_position_size * 100,
            'recommended_shares': int((final_size * portfolio_value) / 100),  # Assuming $100 per share
            'stop_loss_distance': stop_loss_distance,
            'confidence_factor': confidence
        }


class DrawdownMonitor:
    """Monitor portfolio drawdown and implement circuit breakers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.max_drawdown_limit = config.get('max_drawdown_limit', 0.10)  # 10% max
        self.daily_loss_limit = config.get('daily_loss_limit', 0.05)  # 5% daily
        self.peak_portfolio_value = 0.0
        self.daily_start_value = 0.0
        self.drawdown_history = []
        
    def update_portfolio_value(self, current_value: float, is_new_day: bool = False):
        """Update portfolio value and calculate drawdown"""
        
        # Update peak value
        if current_value > self.peak_portfolio_value:
            self.peak_portfolio_value = current_value
            
        # Set daily start value
        if is_new_day:
            self.daily_start_value = current_value
            
        # Calculate drawdowns
        current_drawdown = (self.peak_portfolio_value - current_value) / self.peak_portfolio_value
        daily_drawdown = (self.daily_start_value - current_value) / self.daily_start_value
        
        # Store in history
        self.drawdown_history.append({
            'timestamp': datetime.now(),
            'portfolio_value': current_value,
            'peak_value': self.peak_portfolio_value,
            'current_drawdown': current_drawdown,
            'daily_drawdown': daily_drawdown
        })
        
        # Keep only last 1000 records
        if len(self.drawdown_history) > 1000:
            self.drawdown_history.pop(0)
            
        return current_drawdown, daily_drawdown
        
    def check_circuit_breakers(self, current_drawdown: float, daily_drawdown: float) -> Dict[str, Any]:
        """Check if circuit breakers should be triggered"""
        
        breakers_triggered = []
        
        if current_drawdown >= self.max_drawdown_limit:
            breakers_triggered.append({
                'type': 'max_drawdown',
                'threshold': self.max_drawdown_limit,
                'current': current_drawdown,
                'action': 'halt_all_trading'
            })
            
        if daily_drawdown >= self.daily_loss_limit:
            breakers_triggered.append({
                'type': 'daily_loss',
                'threshold': self.daily_loss_limit,
                'current': daily_drawdown,
                'action': 'halt_new_positions'
            })
            
        return {
            'breakers_triggered': breakers_triggered,
            'current_drawdown': current_drawdown,
            'daily_drawdown': daily_drawdown,
            'trading_allowed': len(breakers_triggered) == 0
        }
        
    def get_max_drawdown_period(self) -> Dict[str, Any]:
        """Get the maximum drawdown period from history"""
        if not self.drawdown_history:
            return {}
            
        max_dd = max(self.drawdown_history, key=lambda x: x['current_drawdown'])
        
        return {
            'max_drawdown': max_dd['current_drawdown'],
            'date': max_dd['timestamp'],
            'portfolio_value_at_max_dd': max_dd['portfolio_value'],
            'peak_value_before_dd': max_dd['peak_value']
        }


class CorrelationAnalyzer:
    """Analyze portfolio correlations and concentration risk"""
    
    def __init__(self, config: Dict[str, Any]):
        self.max_correlation = config.get('max_correlation', 0.8)
        self.lookback_days = config.get('correlation_lookback_days', 252)  # 1 year
        self.price_history = {}
        
    def update_price_data(self, symbol: str, price: float, timestamp: datetime):
        """Update price data for correlation analysis"""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
            
        self.price_history[symbol].append({
            'timestamp': timestamp,
            'price': price
        })
        
        # Keep only required lookback period
        cutoff_date = timestamp - timedelta(days=self.lookback_days)
        self.price_history[symbol] = [
            p for p in self.price_history[symbol] 
            if p['timestamp'] > cutoff_date
        ]
        
    def calculate_correlation_matrix(self, symbols: List[str]) -> np.ndarray:
        """Calculate correlation matrix for given symbols"""
        
        # Get price series for all symbols
        price_series = {}
        min_length = float('inf')
        
        for symbol in symbols:
            if symbol in self.price_history and len(self.price_history[symbol]) > 20:
                prices = [p['price'] for p in self.price_history[symbol]]
                returns = np.diff(np.log(prices))  # Log returns
                price_series[symbol] = returns
                min_length = min(min_length, len(returns))
            else:
                # Not enough data for this symbol
                return np.eye(len(symbols))  # Identity matrix
                
        if not price_series or min_length < 20:
            return np.eye(len(symbols))
            
        # Align series to same length
        aligned_series = {}
        for symbol in symbols:
            if symbol in price_series:
                aligned_series[symbol] = price_series[symbol][-min_length:]
            else:
                # Fill with zeros for missing symbols
                aligned_series[symbol] = np.zeros(min_length)
                
        # Create correlation matrix
        data_matrix = np.array([aligned_series[symbol] for symbol in symbols])
        correlation_matrix = np.corrcoef(data_matrix)
        
        return correlation_matrix
        
    def analyze_portfolio_correlation(self, 
                                    positions: Dict[str, float],
                                    symbols: List[str]) -> Dict[str, Any]:
        """Analyze correlation risk in portfolio"""
        
        if len(symbols) < 2:
            return {
                'correlation_risk': 0.0,
                'max_correlation': 0.0,
                'highly_correlated_pairs': [],
                'correlation_matrix': {}
            }
            
        # Calculate correlation matrix
        corr_matrix = self.calculate_correlation_matrix(symbols)
        
        # Find highly correlated pairs
        highly_correlated = []
        max_correlation = 0.0
        
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                correlation = abs(corr_matrix[i, j])
                if correlation > self.max_correlation:
                    highly_correlated.append({
                        'symbol1': symbols[i],
                        'symbol2': symbols[j],
                        'correlation': correlation,
                        'position1_weight': positions.get(symbols[i], 0),
                        'position2_weight': positions.get(symbols[j], 0)
                    })
                    
                max_correlation = max(max_correlation, correlation)
                
        # Calculate overall correlation risk
        # Weight correlations by position sizes
        weighted_correlation_risk = 0.0
        total_weight = 0.0
        
        for pair in highly_correlated:
            combined_weight = pair['position1_weight'] + pair['position2_weight']
            weighted_correlation_risk += pair['correlation'] * combined_weight
            total_weight += combined_weight
            
        correlation_risk = weighted_correlation_risk / total_weight if total_weight > 0 else 0.0
        
        return {
            'correlation_risk': correlation_risk,
            'max_correlation': max_correlation,
            'highly_correlated_pairs': highly_correlated,
            'correlation_matrix': corr_matrix.tolist(),
            'symbols': symbols
        }


class RiskManagementAgent(BaseAgent):
    """Risk management and compliance agent"""
    
    def __init__(self, message_bus, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.RISK_MANAGEMENT, message_bus, config)
        
        self.position_calculator = PositionSizeCalculator(config or {})
        self.drawdown_monitor = DrawdownMonitor(config or {})
        self.correlation_analyzer = CorrelationAnalyzer(config or {})
        
        # Risk limits
        self.risk_limits = self._initialize_risk_limits(config or {})
        self.active_alerts = []
        self.trading_halted = False
        
        # Circuit breaker status
        self.circuit_breaker_status = {
            'triggered': False,
            'reason': None,
            'triggered_at': None
        }
        
    def _initialize_risk_limits(self, config: Dict[str, Any]) -> List[RiskLimit]:
        """Initialize risk limits from configuration"""
        
        limits = [
            RiskLimit(
                name="Maximum Position Size",
                risk_type=RiskType.POSITION_SIZE,
                limit_value=config.get('max_position_size', 0.25),
                current_value=0.0,
                breach_threshold=config.get('max_position_size', 0.25),
                warning_threshold=config.get('max_position_size', 0.25) * 0.8,
                status=RiskStatus.GREEN,
                description="Maximum percentage of portfolio in single position"
            ),
            RiskLimit(
                name="Maximum Drawdown",
                risk_type=RiskType.DRAWDOWN,
                limit_value=config.get('max_drawdown_limit', 0.10),
                current_value=0.0,
                breach_threshold=config.get('max_drawdown_limit', 0.10),
                warning_threshold=config.get('max_drawdown_limit', 0.10) * 0.8,
                status=RiskStatus.GREEN,
                description="Maximum portfolio drawdown from peak"
            ),
            RiskLimit(
                name="Portfolio Concentration",
                risk_type=RiskType.CONCENTRATION,
                limit_value=config.get('max_concentration', 0.60),
                current_value=0.0,
                breach_threshold=config.get('max_concentration', 0.60),
                warning_threshold=config.get('max_concentration', 0.60) * 0.9,
                status=RiskStatus.GREEN,
                description="Maximum concentration in top 5 positions"
            ),
            RiskLimit(
                name="Daily Loss Limit",
                risk_type=RiskType.DRAWDOWN,
                limit_value=config.get('daily_loss_limit', 0.05),
                current_value=0.0,
                breach_threshold=config.get('daily_loss_limit', 0.05),
                warning_threshold=config.get('daily_loss_limit', 0.05) * 0.8,
                status=RiskStatus.GREEN,
                description="Maximum daily portfolio loss"
            )
        ]
        
        return limits
        
    async def _initialize(self):
        """Initialize the risk management agent"""
        logger.info("Risk Management Agent initialized")
        
    async def _cleanup(self):
        """Cleanup resources"""
        pass
        
    async def process_message(self, message: AgentMessage) -> Optional[AgentResponse]:
        """Process risk management requests"""
        
        if message.message_type == MessageType.RISK_CHECK:
            return await self._handle_risk_check(message)
        elif message.message_type == MessageType.STATUS_UPDATE:
            return await self._handle_status_request(message)
            
        return None
        
    async def _handle_risk_check(self, message: AgentMessage) -> AgentResponse:
        """Handle risk check requests"""
        try:
            check_type = message.payload.get('check_type')
            
            if check_type == 'pre_trade':
                return await self._pre_trade_risk_check(message.payload)
            elif check_type == 'portfolio':
                return await self._portfolio_risk_check(message.payload)
            elif check_type == 'position_size':
                return await self._position_size_check(message.payload)
            else:
                return AgentResponse(
                    success=False,
                    error="Unknown risk check type",
                    confidence=0.0
                )
                
        except Exception as e:
            logger.error(f"Error in risk check: {e}")
            return AgentResponse(
                success=False,
                error=str(e),
                confidence=0.0
            )
            
    async def _handle_status_request(self, message: AgentMessage) -> AgentResponse:
        """Handle status requests"""
        try:
            status_data = {
                'risk_limits': [asdict(limit) for limit in self.risk_limits],
                'active_alerts': [asdict(alert) for alert in self.active_alerts],
                'trading_halted': self.trading_halted,
                'circuit_breaker_status': self.circuit_breaker_status,
                'drawdown_info': self.drawdown_monitor.get_max_drawdown_period()
            }
            
            return AgentResponse(
                success=True,
                data=status_data,
                confidence=1.0
            )
            
        except Exception as e:
            logger.error(f"Error getting risk status: {e}")
            return AgentResponse(
                success=False,
                error=str(e),
                confidence=0.0
            )
            
    async def _pre_trade_risk_check(self, payload: Dict[str, Any]) -> AgentResponse:
        """Perform pre-trade risk checks"""
        
        trading_decision = payload.get('trading_decision', {})
        portfolio_data = payload.get('portfolio_data', {})
        
        risk_checks = []
        overall_approved = True
        
        # Check position size limits
        position_size_pct = trading_decision.get('position_size_pct', 0)
        max_position_limit = self._get_risk_limit(RiskType.POSITION_SIZE)
        
        if position_size_pct > max_position_limit.breach_threshold * 100:
            overall_approved = False
            risk_checks.append({
                'check': 'position_size',
                'status': 'FAILED',
                'message': f"Position size {position_size_pct}% exceeds limit {max_position_limit.breach_threshold*100}%"
            })
        else:
            risk_checks.append({
                'check': 'position_size',
                'status': 'PASSED',
                'message': f"Position size {position_size_pct}% within limits"
            })
            
        # Check if trading is halted
        if self.trading_halted:
            overall_approved = False
            risk_checks.append({
                'check': 'trading_halt',
                'status': 'FAILED',
                'message': f"Trading halted: {self.circuit_breaker_status.get('reason', 'Unknown')}"
            })
        else:
            risk_checks.append({
                'check': 'trading_halt',
                'status': 'PASSED',
                'message': "Trading is allowed"
            })
            
        # Check portfolio concentration (simplified)
        current_positions = portfolio_data.get('positions', [])
        if current_positions:
            total_value = sum(pos.get('market_value', 0) for pos in current_positions)
            if total_value > 0:
                # Calculate concentration in top positions
                sorted_positions = sorted(current_positions, key=lambda x: x.get('market_value', 0), reverse=True)
                top_5_value = sum(pos.get('market_value', 0) for pos in sorted_positions[:5])
                concentration = top_5_value / total_value
                
                concentration_limit = self._get_risk_limit(RiskType.CONCENTRATION)
                if concentration > concentration_limit.breach_threshold:
                    overall_approved = False
                    risk_checks.append({
                        'check': 'concentration',
                        'status': 'FAILED',
                        'message': f"Portfolio concentration {concentration:.1%} exceeds limit {concentration_limit.breach_threshold:.1%}"
                    })
                else:
                    risk_checks.append({
                        'check': 'concentration',
                        'status': 'PASSED',
                        'message': f"Portfolio concentration {concentration:.1%} within limits"
                    })
                    
        return AgentResponse(
            success=True,
            data={
                'approved': overall_approved,
                'risk_checks': risk_checks,
                'recommended_adjustments': self._get_recommended_adjustments(trading_decision, risk_checks)
            },
            confidence=1.0
        )
        
    async def _portfolio_risk_check(self, payload: Dict[str, Any]) -> AgentResponse:
        """Perform comprehensive portfolio risk analysis"""
        
        portfolio_data = payload.get('portfolio_data', {})
        
        # Update drawdown monitoring
        current_value = portfolio_data.get('total_value', 0)
        is_new_day = payload.get('is_new_day', False)
        
        current_drawdown, daily_drawdown = self.drawdown_monitor.update_portfolio_value(
            current_value, is_new_day
        )
        
        # Check circuit breakers
        circuit_breaker_check = self.drawdown_monitor.check_circuit_breakers(
            current_drawdown, daily_drawdown
        )
        
        # Update trading halt status
        if circuit_breaker_check['breakers_triggered']:
            self.trading_halted = True
            self.circuit_breaker_status = {
                'triggered': True,
                'reason': f"Circuit breakers triggered: {[b['type'] for b in circuit_breaker_check['breakers_triggered']]}",
                'triggered_at': datetime.now()
            }
            
            # Create risk alert
            self._create_risk_alert(
                RiskType.DRAWDOWN,
                RiskLevel.EXTREME,
                f"Circuit breaker triggered: {circuit_breaker_check['breakers_triggered'][0]['type']}",
                "Halt all trading immediately"
            )
            
        # Analyze portfolio correlations
        positions = portfolio_data.get('positions', [])
        symbols = [pos.get('symbol') for pos in positions if pos.get('symbol')]
        position_weights = {pos.get('symbol'): pos.get('market_value', 0) / current_value 
                          for pos in positions if pos.get('symbol')}
        
        correlation_analysis = self.correlation_analyzer.analyze_portfolio_correlation(
            position_weights, symbols
        )
        
        # Update risk limits with current values
        self._update_risk_limits(current_drawdown, daily_drawdown, position_weights, correlation_analysis)
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(portfolio_data, correlation_analysis)
        
        return AgentResponse(
            success=True,
            data={
                'risk_metrics': asdict(risk_metrics),
                'circuit_breakers': circuit_breaker_check,
                'correlation_analysis': correlation_analysis,
                'risk_limits_status': [asdict(limit) for limit in self.risk_limits],
                'trading_status': 'HALTED' if self.trading_halted else 'ACTIVE'
            },
            confidence=1.0
        )
        
    async def _position_size_check(self, payload: Dict[str, Any]) -> AgentResponse:
        """Calculate optimal position size based on risk parameters"""
        
        portfolio_value = payload.get('portfolio_value', 0)
        confidence = payload.get('confidence', 0.5)
        stop_loss_price = payload.get('stop_loss_price')
        current_price = payload.get('current_price', 100)
        volatility = payload.get('volatility')
        
        # Calculate stop loss distance
        stop_loss_distance = 0
        if stop_loss_price and current_price:
            stop_loss_distance = abs(current_price - stop_loss_price)
            
        # Calculate position size
        position_calc = self.position_calculator.calculate_position_size(
            portfolio_value, confidence, stop_loss_distance, volatility
        )
        
        return AgentResponse(
            success=True,
            data=position_calc,
            confidence=1.0
        )
        
    def _get_risk_limit(self, risk_type: RiskType) -> RiskLimit:
        """Get risk limit by type"""
        for limit in self.risk_limits:
            if limit.risk_type == risk_type:
                return limit
        # Return default if not found
        return RiskLimit(
            name="Default",
            risk_type=risk_type,
            limit_value=0.25,
            current_value=0.0,
            breach_threshold=0.25,
            warning_threshold=0.20,
            status=RiskStatus.GREEN,
            description="Default risk limit"
        )
        
    def _update_risk_limits(self, 
                           current_drawdown: float,
                           daily_drawdown: float,
                           position_weights: Dict[str, float],
                           correlation_analysis: Dict[str, Any]):
        """Update risk limits with current values"""
        
        for limit in self.risk_limits:
            if limit.risk_type == RiskType.DRAWDOWN:
                if "daily" in limit.name.lower():
                    limit.current_value = daily_drawdown
                else:
                    limit.current_value = current_drawdown
                    
            elif limit.risk_type == RiskType.POSITION_SIZE:
                if position_weights:
                    limit.current_value = max(position_weights.values())
                    
            elif limit.risk_type == RiskType.CONCENTRATION:
                if position_weights:
                    # Top 5 positions concentration
                    sorted_weights = sorted(position_weights.values(), reverse=True)
                    limit.current_value = sum(sorted_weights[:5])
                    
            # Update status based on current value
            if limit.current_value >= limit.breach_threshold:
                limit.status = RiskStatus.RED
            elif limit.current_value >= limit.warning_threshold:
                limit.status = RiskStatus.YELLOW
            else:
                limit.status = RiskStatus.GREEN
                
    def _calculate_risk_metrics(self, 
                               portfolio_data: Dict[str, Any],
                               correlation_analysis: Dict[str, Any]) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        
        positions = portfolio_data.get('positions', [])
        total_value = portfolio_data.get('total_value', 0)
        cash = portfolio_data.get('cash', 0)
        
        # Calculate metrics
        total_exposure = sum(pos.get('market_value', 0) for pos in positions)
        largest_position_pct = max([pos.get('market_value', 0) / total_value for pos in positions]) if positions and total_value > 0 else 0
        cash_ratio = cash / total_value if total_value > 0 else 1.0
        
        # Get max drawdown from monitor
        max_dd_info = self.drawdown_monitor.get_max_drawdown_period()
        max_drawdown = max_dd_info.get('max_drawdown', 0)
        
        # Current drawdown
        current_drawdown = 0
        if self.drawdown_monitor.drawdown_history:
            current_drawdown = self.drawdown_monitor.drawdown_history[-1]['current_drawdown']
            
        return RiskMetrics(
            total_exposure=total_exposure,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            portfolio_var=total_value * 0.05,  # Simplified 5% VaR
            portfolio_volatility=0.15,  # Would calculate from returns
            sharpe_ratio=1.2,  # Would calculate from returns
            beta=1.0,  # Would calculate vs benchmark
            largest_position_pct=largest_position_pct,
            correlation_risk=correlation_analysis.get('correlation_risk', 0),
            cash_ratio=cash_ratio,
            updated_at=datetime.now()
        )
        
    def _get_recommended_adjustments(self, 
                                   trading_decision: Dict[str, Any],
                                   risk_checks: List[Dict[str, Any]]) -> List[str]:
        """Get recommended adjustments based on risk checks"""
        
        adjustments = []
        
        for check in risk_checks:
            if check['status'] == 'FAILED':
                if check['check'] == 'position_size':
                    max_limit = self._get_risk_limit(RiskType.POSITION_SIZE)
                    adjustments.append(f"Reduce position size to maximum {max_limit.breach_threshold*100:.1f}%")
                elif check['check'] == 'concentration':
                    adjustments.append("Reduce concentration by diversifying into additional positions")
                elif check['check'] == 'trading_halt':
                    adjustments.append("Wait for trading halt to be lifted before proceeding")
                    
        return adjustments
        
    def _create_risk_alert(self, 
                          risk_type: RiskType,
                          severity: RiskLevel,
                          message: str,
                          recommended_action: str,
                          affected_symbols: List[str] = None) -> RiskAlert:
        """Create a new risk alert"""
        
        alert = RiskAlert(
            alert_id=str(uuid.uuid4()),
            risk_type=risk_type,
            severity=severity,
            message=message,
            recommended_action=recommended_action,
            affected_symbols=affected_symbols or [],
            created_at=datetime.now()
        )
        
        self.active_alerts.append(alert)
        
        # Keep only recent alerts (last 100)
        if len(self.active_alerts) > 100:
            self.active_alerts.pop(0)
            
        logger.warning(f"Risk Alert Created: {message}")
        
        return alert