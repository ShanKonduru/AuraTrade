import asyncio
import uuid
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from alpaca.data.historical import StockHistoricalDataClient
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
from loguru import logger

from ..base_agent import BaseAgent, AgentMessage, AgentResponse
from ..agent_types import AgentType, MessageType, ActionType, RiskLevel


class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class BrokerType(Enum):
    """Supported broker types"""
    ALPACA = "alpaca"
    INTERACTIVE_BROKERS = "interactive_brokers"
    PAPER_TRADING = "paper_trading"


@dataclass
class Order:
    """Order structure"""
    order_id: str
    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    order_type: str  # "market", "limit", "stop"
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "day"
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = None
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None
    filled_quantity: Optional[float] = None
    commission: Optional[float] = None
    broker_order_id: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class Position:
    """Position structure"""
    symbol: str
    quantity: float
    average_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    side: str  # "long" or "short"
    created_at: datetime


@dataclass
class Portfolio:
    """Portfolio structure"""
    total_value: float
    cash: float
    buying_power: float
    positions: List[Position]
    day_pnl: float
    total_pnl: float
    updated_at: datetime


class BaseBroker:
    """Base broker interface"""
    
    async def connect(self) -> bool:
        """Connect to broker"""
        raise NotImplementedError
        
    async def disconnect(self):
        """Disconnect from broker"""
        raise NotImplementedError
        
    async def submit_order(self, order: Order) -> Order:
        """Submit order to broker"""
        raise NotImplementedError
        
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order"""
        raise NotImplementedError
        
    async def get_order_status(self, order_id: str) -> Order:
        """Get order status"""
        raise NotImplementedError
        
    async def get_portfolio(self) -> Portfolio:
        """Get current portfolio"""
        raise NotImplementedError
        
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol"""
        raise NotImplementedError
        
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        raise NotImplementedError


class AlpacaBroker(BaseBroker):
    """Alpaca broker implementation"""
    
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper
        self.trading_client = None
        self.data_client = None
        self.connected = False
        
    async def connect(self) -> bool:
        """Connect to Alpaca"""
        try:
            self.trading_client = TradingClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
                paper=self.paper
            )
            
            self.data_client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.secret_key
            )
            
            # Test connection
            account = await asyncio.to_thread(self.trading_client.get_account)
            self.connected = True
            logger.info(f"Connected to Alpaca {'paper' if self.paper else 'live'} trading")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            self.connected = False
            return False
            
    async def disconnect(self):
        """Disconnect from Alpaca"""
        self.connected = False
        logger.info("Disconnected from Alpaca")
        
    async def submit_order(self, order: Order) -> Order:
        """Submit order to Alpaca"""
        if not self.connected:
            raise Exception("Not connected to broker")
            
        try:
            # Convert order to Alpaca format
            side = OrderSide.BUY if order.side.lower() == "buy" else OrderSide.SELL
            
            if order.order_type.lower() == "market":
                order_request = MarketOrderRequest(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY
                )
            elif order.order_type.lower() == "limit":
                order_request = LimitOrderRequest(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=order.price
                )
            elif order.order_type.lower() == "stop":
                order_request = StopOrderRequest(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    stop_price=order.stop_price
                )
            else:
                raise ValueError(f"Unsupported order type: {order.order_type}")
                
            # Submit order
            alpaca_order = await asyncio.to_thread(
                self.trading_client.submit_order, order_request
            )
            
            # Update order with broker response
            order.broker_order_id = alpaca_order.id
            order.status = OrderStatus.SUBMITTED
            
            logger.info(f"Order submitted: {order.order_id} -> {alpaca_order.id}")
            return order
            
        except Exception as e:
            logger.error(f"Failed to submit order: {e}")
            order.status = OrderStatus.REJECTED
            return order
            
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order with Alpaca"""
        if not self.connected:
            return False
            
        try:
            await asyncio.to_thread(self.trading_client.cancel_order_by_id, order_id)
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
            
    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get order status from Alpaca"""
        if not self.connected:
            return None
            
        try:
            alpaca_order = await asyncio.to_thread(
                self.trading_client.get_order_by_id, order_id
            )
            
            # Convert Alpaca order to our Order format
            order = Order(
                order_id=str(uuid.uuid4()),  # Our internal ID
                symbol=alpaca_order.symbol,
                side=alpaca_order.side.value.lower(),
                quantity=float(alpaca_order.qty),
                order_type=alpaca_order.order_type.value.lower(),
                price=float(alpaca_order.limit_price) if alpaca_order.limit_price else None,
                stop_price=float(alpaca_order.stop_price) if alpaca_order.stop_price else None,
                status=self._convert_alpaca_status(alpaca_order.status),
                broker_order_id=alpaca_order.id,
                filled_price=float(alpaca_order.filled_avg_price) if alpaca_order.filled_avg_price else None,
                filled_quantity=float(alpaca_order.filled_qty) if alpaca_order.filled_qty else None
            )
            
            return order
            
        except Exception as e:
            logger.error(f"Failed to get order status {order_id}: {e}")
            return None
            
    def _convert_alpaca_status(self, alpaca_status) -> OrderStatus:
        """Convert Alpaca order status to our status"""
        status_map = {
            'new': OrderStatus.SUBMITTED,
            'partially_filled': OrderStatus.PARTIALLY_FILLED,
            'filled': OrderStatus.FILLED,
            'done_for_day': OrderStatus.CANCELLED,
            'canceled': OrderStatus.CANCELLED,
            'expired': OrderStatus.EXPIRED,
            'replaced': OrderStatus.SUBMITTED,
            'pending_cancel': OrderStatus.SUBMITTED,
            'pending_replace': OrderStatus.SUBMITTED,
            'accepted': OrderStatus.SUBMITTED,
            'pending_new': OrderStatus.PENDING,
            'accepted_for_bidding': OrderStatus.SUBMITTED,
            'stopped': OrderStatus.CANCELLED,
            'rejected': OrderStatus.REJECTED,
            'suspended': OrderStatus.CANCELLED
        }
        
        return status_map.get(alpaca_status.value.lower(), OrderStatus.PENDING)
        
    async def get_portfolio(self) -> Portfolio:
        """Get portfolio from Alpaca"""
        if not self.connected:
            raise Exception("Not connected to broker")
            
        try:
            # Get account info
            account = await asyncio.to_thread(self.trading_client.get_account)
            
            # Get positions
            alpaca_positions = await asyncio.to_thread(self.trading_client.get_all_positions)
            
            positions = []
            for alpaca_pos in alpaca_positions:
                position = Position(
                    symbol=alpaca_pos.symbol,
                    quantity=float(alpaca_pos.qty),
                    average_cost=float(alpaca_pos.avg_entry_price),
                    current_price=float(alpaca_pos.current_price),
                    market_value=float(alpaca_pos.market_value),
                    unrealized_pnl=float(alpaca_pos.unrealized_pl),
                    realized_pnl=0.0,  # Would need to calculate from trades
                    side="long" if float(alpaca_pos.qty) > 0 else "short",
                    created_at=datetime.now()  # Would need actual entry date
                )
                positions.append(position)
                
            portfolio = Portfolio(
                total_value=float(account.portfolio_value),
                cash=float(account.cash),
                buying_power=float(account.buying_power),
                positions=positions,
                day_pnl=float(account.daytrading_buying_power),
                total_pnl=sum(pos.unrealized_pnl for pos in positions),
                updated_at=datetime.now()
            )
            
            return portfolio
            
        except Exception as e:
            logger.error(f"Failed to get portfolio: {e}")
            raise
            
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for specific symbol"""
        try:
            portfolio = await self.get_portfolio()
            for position in portfolio.positions:
                if position.symbol == symbol:
                    return position
            return None
        except Exception as e:
            logger.error(f"Failed to get position for {symbol}: {e}")
            return None
            
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        if not self.connected:
            raise Exception("Not connected to broker")
            
        try:
            account = await asyncio.to_thread(self.trading_client.get_account)
            return {
                'account_id': account.id,
                'status': account.status,
                'currency': account.currency,
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'buying_power': float(account.buying_power),
                'daytrading_buying_power': float(account.daytrading_buying_power),
                'pattern_day_trader': account.pattern_day_trader,
                'trading_blocked': account.trading_blocked,
                'transfers_blocked': account.transfers_blocked
            }
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            raise


class PaperTradingBroker(BaseBroker):
    """Paper trading broker for simulation"""
    
    def __init__(self, initial_cash: float = 100000.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = {}
        self.orders = {}
        self.order_history = []
        self.connected = False
        
    async def connect(self) -> bool:
        """Connect to paper trading"""
        self.connected = True
        logger.info("Connected to paper trading")
        return True
        
    async def disconnect(self):
        """Disconnect from paper trading"""
        self.connected = False
        logger.info("Disconnected from paper trading")
        
    async def submit_order(self, order: Order) -> Order:
        """Submit order to paper trading"""
        if not self.connected:
            raise Exception("Not connected to broker")
            
        # Simulate order execution
        order.status = OrderStatus.SUBMITTED
        order.broker_order_id = order.order_id
        
        # For market orders, execute immediately
        if order.order_type.lower() == "market":
            await self._execute_order(order)
            
        self.orders[order.order_id] = order
        logger.info(f"Paper order submitted: {order.order_id}")
        return order
        
    async def _execute_order(self, order: Order):
        """Execute order in paper trading"""
        # Get current price (simplified - would use real market data)
        current_price = 100.0  # This would come from real-time data
        
        order.filled_price = current_price
        order.filled_quantity = order.quantity
        order.filled_at = datetime.now()
        order.status = OrderStatus.FILLED
        
        # Update positions and cash
        if order.side.lower() == "buy":
            cost = order.quantity * current_price
            if cost <= self.cash:
                self.cash -= cost
                if order.symbol in self.positions:
                    # Update existing position
                    pos = self.positions[order.symbol]
                    total_quantity = pos.quantity + order.quantity
                    total_cost = (pos.quantity * pos.average_cost) + cost
                    pos.average_cost = total_cost / total_quantity
                    pos.quantity = total_quantity
                else:
                    # Create new position
                    self.positions[order.symbol] = Position(
                        symbol=order.symbol,
                        quantity=order.quantity,
                        average_cost=current_price,
                        current_price=current_price,
                        market_value=order.quantity * current_price,
                        unrealized_pnl=0.0,
                        realized_pnl=0.0,
                        side="long",
                        created_at=datetime.now()
                    )
            else:
                order.status = OrderStatus.REJECTED
                logger.warning(f"Insufficient funds for order {order.order_id}")
                
        elif order.side.lower() == "sell":
            if order.symbol in self.positions:
                pos = self.positions[order.symbol]
                if pos.quantity >= order.quantity:
                    # Execute sell
                    proceeds = order.quantity * current_price
                    self.cash += proceeds
                    pos.quantity -= order.quantity
                    
                    # Calculate realized P&L
                    realized_pnl = (current_price - pos.average_cost) * order.quantity
                    pos.realized_pnl += realized_pnl
                    
                    # Remove position if quantity is zero
                    if pos.quantity == 0:
                        del self.positions[order.symbol]
                else:
                    order.status = OrderStatus.REJECTED
                    logger.warning(f"Insufficient shares for sell order {order.order_id}")
            else:
                order.status = OrderStatus.REJECTED
                logger.warning(f"No position to sell for {order.symbol}")
                
        self.order_history.append(order)
        
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order in paper trading"""
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED]:
                order.status = OrderStatus.CANCELLED
                logger.info(f"Paper order cancelled: {order_id}")
                return True
        return False
        
    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get order status in paper trading"""
        return self.orders.get(order_id)
        
    async def get_portfolio(self) -> Portfolio:
        """Get paper trading portfolio"""
        total_value = self.cash
        day_pnl = 0.0
        total_pnl = 0.0
        
        positions = []
        for symbol, pos in self.positions.items():
            # Update current price and P&L (simplified)
            pos.current_price = 100.0  # Would use real market data
            pos.market_value = pos.quantity * pos.current_price
            pos.unrealized_pnl = (pos.current_price - pos.average_cost) * pos.quantity
            
            total_value += pos.market_value
            total_pnl += pos.unrealized_pnl + pos.realized_pnl
            
            positions.append(pos)
            
        return Portfolio(
            total_value=total_value,
            cash=self.cash,
            buying_power=self.cash,  # Simplified
            positions=positions,
            day_pnl=day_pnl,
            total_pnl=total_pnl,
            updated_at=datetime.now()
        )
        
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol in paper trading"""
        return self.positions.get(symbol)
        
    async def get_account_info(self) -> Dict[str, Any]:
        """Get paper trading account info"""
        portfolio = await self.get_portfolio()
        return {
            'account_id': 'paper_trading',
            'status': 'ACTIVE',
            'currency': 'USD',
            'cash': self.cash,
            'portfolio_value': portfolio.total_value,
            'buying_power': self.cash,
            'pattern_day_trader': False,
            'trading_blocked': False,
            'transfers_blocked': False
        }


class ExecutionAgent(BaseAgent):
    """Agent for trade execution and order management"""
    
    def __init__(self, message_bus, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.EXECUTION, message_bus, config)
        
        self.broker = None
        self.active_orders = {}
        self.execution_history = []
        
        # Configuration
        self.broker_type = config.get('broker_type', BrokerType.PAPER_TRADING)
        self.max_position_size = config.get('max_position_size', 0.25)  # 25% max position
        self.slippage_tolerance = config.get('slippage_tolerance', 0.001)  # 0.1%
        
    async def _initialize(self):
        """Initialize the execution agent"""
        await self._initialize_broker()
        logger.info("Execution Agent initialized")
        
    async def _cleanup(self):
        """Cleanup resources"""
        if self.broker:
            await self.broker.disconnect()
            
    async def _initialize_broker(self):
        """Initialize broker connection"""
        if self.broker_type == BrokerType.ALPACA:
            api_key = self.config.get('alpaca_api_key')
            secret_key = self.config.get('alpaca_secret_key')
            paper = self.config.get('alpaca_paper', True)
            
            if api_key and secret_key:
                self.broker = AlpacaBroker(api_key, secret_key, paper)
            else:
                logger.warning("Alpaca credentials not provided, using paper trading")
                self.broker = PaperTradingBroker()
        else:
            # Default to paper trading
            initial_cash = self.config.get('initial_cash', 100000.0)
            self.broker = PaperTradingBroker(initial_cash)
            
        # Connect to broker
        connected = await self.broker.connect()
        if not connected:
            raise Exception("Failed to connect to broker")
            
    async def process_message(self, message: AgentMessage) -> Optional[AgentResponse]:
        """Process execution requests"""
        
        if message.message_type == MessageType.TRADE_EXECUTION:
            return await self._handle_trade_execution(message)
        elif message.message_type == MessageType.STATUS_UPDATE:
            return await self._handle_status_request(message)
            
        return None
        
    async def _handle_trade_execution(self, message: AgentMessage) -> AgentResponse:
        """Handle trade execution requests"""
        try:
            trading_decision = message.payload
            
            # Validate trading decision
            if not self._validate_trading_decision(trading_decision):
                return AgentResponse(
                    success=False,
                    error="Invalid trading decision",
                    confidence=0.0
                )
                
            # Execute trade
            execution_result = await self._execute_trade(trading_decision)
            
            return AgentResponse(
                success=execution_result['success'],
                data=execution_result,
                confidence=1.0 if execution_result['success'] else 0.0,
                metadata={'symbol': trading_decision.get('symbol')}
            )
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return AgentResponse(
                success=False,
                error=str(e),
                confidence=0.0
            )
            
    async def _handle_status_request(self, message: AgentMessage) -> AgentResponse:
        """Handle status requests"""
        try:
            portfolio = await self.broker.get_portfolio()
            account_info = await self.broker.get_account_info()
            
            status_data = {
                'portfolio': asdict(portfolio),
                'account_info': account_info,
                'active_orders': len(self.active_orders),
                'execution_history': len(self.execution_history)
            }
            
            return AgentResponse(
                success=True,
                data=status_data,
                confidence=1.0
            )
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return AgentResponse(
                success=False,
                error=str(e),
                confidence=0.0
            )
            
    def _validate_trading_decision(self, trading_decision: Dict[str, Any]) -> bool:
        """Validate trading decision before execution"""
        required_fields = ['symbol', 'action', 'position_size_pct', 'confidence']
        
        for field in required_fields:
            if field not in trading_decision:
                logger.error(f"Missing required field: {field}")
                return False
                
        # Validate position size
        position_size = trading_decision.get('position_size_pct', 0) / 100.0
        if position_size > self.max_position_size:
            logger.error(f"Position size {position_size} exceeds maximum {self.max_position_size}")
            return False
            
        # Validate action
        action = trading_decision.get('action')
        if action not in [a.value for a in ActionType]:
            logger.error(f"Invalid action: {action}")
            return False
            
        return True
        
    async def _execute_trade(self, trading_decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute trading decision"""
        
        symbol = trading_decision['symbol']
        action = ActionType(trading_decision['action'])
        position_size_pct = trading_decision['position_size_pct'] / 100.0
        confidence = trading_decision['confidence']
        
        try:
            # Get current portfolio
            portfolio = await self.broker.get_portfolio()
            current_position = await self.broker.get_position(symbol)
            
            # Calculate order quantity
            order_quantity = await self._calculate_order_quantity(
                portfolio, current_position, action, position_size_pct
            )
            
            if order_quantity == 0:
                return {
                    'success': True,
                    'message': 'No trade required',
                    'action': 'HOLD',
                    'quantity': 0
                }
                
            # Determine order side
            order_side = self._determine_order_side(action, current_position)
            
            # Create order
            order = Order(
                order_id=str(uuid.uuid4()),
                symbol=symbol,
                side=order_side,
                quantity=abs(order_quantity),
                order_type="market",  # Could be configurable
                time_in_force="day"
            )
            
            # Submit order
            submitted_order = await self.broker.submit_order(order)
            
            # Track order
            self.active_orders[submitted_order.order_id] = submitted_order
            
            # Add to execution history
            execution_record = {
                'timestamp': datetime.now().isoformat(),
                'trading_decision': trading_decision,
                'order': asdict(submitted_order),
                'portfolio_value_before': portfolio.total_value
            }
            self.execution_history.append(execution_record)
            
            return {
                'success': True,
                'order_id': submitted_order.order_id,
                'symbol': symbol,
                'action': action.value,
                'quantity': order_quantity,
                'order_type': 'market',
                'status': submitted_order.status.value,
                'message': f"Order submitted for {symbol}: {action.value} {abs(order_quantity)} shares"
            }
            
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
            return {
                'success': False,
                'error': str(e),
                'symbol': symbol,
                'action': action.value
            }
            
    async def _calculate_order_quantity(self, 
                                      portfolio: Portfolio,
                                      current_position: Optional[Position],
                                      action: ActionType,
                                      position_size_pct: float) -> float:
        """Calculate order quantity based on action and position sizing"""
        
        # Get current price (simplified - would use real market data)
        current_price = 100.0  # This would come from market data
        
        # Calculate target position size in dollars
        target_position_value = portfolio.total_value * position_size_pct
        target_quantity = target_position_value / current_price
        
        current_quantity = current_position.quantity if current_position else 0.0
        
        if action in [ActionType.BUY, ActionType.ACCUMULATE]:
            # Calculate quantity to buy
            if action == ActionType.ACCUMULATE:
                # For accumulate, increase position size
                return target_quantity - current_quantity
            else:
                # For buy, establish position if none exists
                if current_quantity == 0:
                    return target_quantity
                else:
                    return 0  # Already have position
                    
        elif action in [ActionType.SELL, ActionType.DIVEST]:
            # Calculate quantity to sell
            if current_quantity > 0:
                if action == ActionType.DIVEST:
                    # Sell entire position
                    return -current_quantity
                else:
                    # Sell portion of position
                    sell_quantity = min(target_quantity, current_quantity)
                    return -sell_quantity
            else:
                return 0  # No position to sell
                
        else:  # HOLD
            return 0
            
    def _determine_order_side(self, action: ActionType, current_position: Optional[Position]) -> str:
        """Determine order side (buy/sell) based on action"""
        
        if action in [ActionType.BUY, ActionType.ACCUMULATE]:
            return "buy"
        elif action in [ActionType.SELL, ActionType.DIVEST]:
            return "sell"
        else:
            return "hold"  # This shouldn't happen in practice