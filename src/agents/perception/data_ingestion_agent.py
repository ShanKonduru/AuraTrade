import asyncio
import aiohttp
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import json
from loguru import logger

from ..base_agent import BaseAgent, AgentMessage, AgentResponse
from ..agent_types import AgentType, MessageType


@dataclass
class MarketData:
    """Market data structure"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    adjusted_close: Optional[float] = None


@dataclass
class FundamentalData:
    """Fundamental data structure"""
    symbol: str
    quarter: str
    revenue: Optional[float]
    net_income: Optional[float]
    eps: Optional[float]
    pe_ratio: Optional[float]
    pb_ratio: Optional[float]
    debt_to_equity: Optional[float]
    roe: Optional[float]
    roa: Optional[float]
    free_cash_flow: Optional[float]


class DataProvider:
    """Base class for data providers"""
    
    async def get_historical_data(self, symbol: str, period: str) -> List[MarketData]:
        raise NotImplementedError
        
    async def get_realtime_data(self, symbol: str) -> MarketData:
        raise NotImplementedError
        
    async def get_fundamental_data(self, symbol: str) -> FundamentalData:
        raise NotImplementedError


class YahooFinanceProvider(DataProvider):
    """Yahoo Finance data provider"""
    
    def __init__(self):
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    async def get_historical_data(self, symbol: str, period: str = "1y") -> List[MarketData]:
        """Get historical market data"""
        try:
            # Use yfinance in a thread to avoid blocking
            ticker = yf.Ticker(symbol)
            df = await asyncio.to_thread(ticker.history, period=period)
            
            market_data = []
            for index, row in df.iterrows():
                market_data.append(MarketData(
                    symbol=symbol,
                    timestamp=index.to_pydatetime(),
                    open=float(row['Open']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    close=float(row['Close']),
                    volume=int(row['Volume']),
                    adjusted_close=float(row.get('Adj Close', row['Close']))
                ))
                
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return []
            
    async def get_realtime_data(self, symbol: str) -> Optional[MarketData]:
        """Get real-time market data"""
        try:
            ticker = yf.Ticker(symbol)
            info = await asyncio.to_thread(lambda: ticker.info)
            hist = await asyncio.to_thread(lambda: ticker.history(period="1d"))
            
            if hist.empty:
                return None
                
            latest = hist.iloc[-1]
            return MarketData(
                symbol=symbol,
                timestamp=datetime.now(),
                open=float(latest['Open']),
                high=float(latest['High']),
                low=float(latest['Low']),
                close=float(latest['Close']),
                volume=int(latest['Volume']),
                adjusted_close=float(info.get('regularMarketPrice', latest['Close']))
            )
            
        except Exception as e:
            logger.error(f"Error fetching real-time data for {symbol}: {e}")
            return None
            
    async def get_fundamental_data(self, symbol: str) -> Optional[FundamentalData]:
        """Get fundamental data"""
        try:
            ticker = yf.Ticker(symbol)
            info = await asyncio.to_thread(lambda: ticker.info)
            
            return FundamentalData(
                symbol=symbol,
                quarter="current",
                revenue=info.get('totalRevenue'),
                net_income=info.get('netIncomeToCommon'),
                eps=info.get('trailingEps'),
                pe_ratio=info.get('trailingPE'),
                pb_ratio=info.get('priceToBook'),
                debt_to_equity=info.get('debtToEquity'),
                roe=info.get('returnOnEquity'),
                roa=info.get('returnOnAssets'),
                free_cash_flow=info.get('freeCashflow')
            )
            
        except Exception as e:
            logger.error(f"Error fetching fundamental data for {symbol}: {e}")
            return None


class DataIngestionAgent(BaseAgent):
    """Agent responsible for collecting and cleaning financial data"""
    
    def __init__(self, message_bus, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.DATA_INGESTION, message_bus, config)
        
        self.data_providers = {}
        self.cache = {}  # Simple in-memory cache
        self.cache_ttl = config.get('cache_ttl', 300)  # 5 minutes default
        
    async def _initialize(self):
        """Initialize data providers"""
        # Initialize Yahoo Finance provider
        self.data_providers['yahoo'] = YahooFinanceProvider()
        
        # Initialize other providers as needed
        # self.data_providers['alpha_vantage'] = AlphaVantageProvider()
        # self.data_providers['polygon'] = PolygonProvider()
        
        logger.info("Data Ingestion Agent initialized with providers")
        
    async def _cleanup(self):
        """Cleanup resources"""
        for provider in self.data_providers.values():
            if hasattr(provider, '__aexit__'):
                await provider.__aexit__(None, None, None)
                
    async def process_message(self, message: AgentMessage) -> Optional[AgentResponse]:
        """Process incoming data requests"""
        
        if message.message_type == MessageType.DATA_REQUEST:
            return await self._handle_data_request(message)
        elif message.message_type == MessageType.STATUS_UPDATE:
            return await self._handle_status_request(message)
            
        return None
        
    async def _handle_data_request(self, message: AgentMessage) -> AgentResponse:
        """Handle data request messages"""
        try:
            request_type = message.payload.get('request_type')
            symbol = message.payload.get('symbol')
            
            if not symbol:
                return AgentResponse(
                    success=False,
                    error="Symbol is required for data requests",
                    confidence=0.0
                )
                
            # Check cache first
            cache_key = f"{request_type}_{symbol}"
            if cache_key in self.cache:
                cached_data, cache_time = self.cache[cache_key]
                if datetime.now() - cache_time < timedelta(seconds=self.cache_ttl):
                    return AgentResponse(
                        success=True,
                        data=cached_data,
                        confidence=1.0,
                        metadata={'source': 'cache'}
                    )
                    
            # Fetch new data
            if request_type == 'historical':
                period = message.payload.get('period', '1y')
                data = await self._get_historical_data(symbol, period)
            elif request_type == 'realtime':
                data = await self._get_realtime_data(symbol)
            elif request_type == 'fundamental':
                data = await self._get_fundamental_data(symbol)
            else:
                return AgentResponse(
                    success=False,
                    error=f"Unknown request type: {request_type}",
                    confidence=0.0
                )
                
            if data:
                # Cache the data
                self.cache[cache_key] = (data, datetime.now())
                
                return AgentResponse(
                    success=True,
                    data=data,
                    confidence=1.0,
                    metadata={'source': 'live', 'timestamp': datetime.now().isoformat()}
                )
            else:
                return AgentResponse(
                    success=False,
                    error=f"Failed to fetch data for {symbol}",
                    confidence=0.0
                )
                
        except Exception as e:
            logger.error(f"Error handling data request: {e}")
            return AgentResponse(
                success=False,
                error=str(e),
                confidence=0.0
            )
            
    async def _get_historical_data(self, symbol: str, period: str) -> Dict[str, Any]:
        """Get historical market data"""
        provider = self.data_providers.get('yahoo')
        if not provider:
            return {}
            
        async with provider:
            market_data = await provider.get_historical_data(symbol, period)
            
        # Convert to serializable format
        return {
            'symbol': symbol,
            'period': period,
            'data': [
                {
                    'timestamp': data.timestamp.isoformat(),
                    'open': data.open,
                    'high': data.high,
                    'low': data.low,
                    'close': data.close,
                    'volume': data.volume,
                    'adjusted_close': data.adjusted_close
                }
                for data in market_data
            ]
        }
        
    async def _get_realtime_data(self, symbol: str) -> Dict[str, Any]:
        """Get real-time market data"""
        provider = self.data_providers.get('yahoo')
        if not provider:
            return {}
            
        async with provider:
            market_data = await provider.get_realtime_data(symbol)
            
        if not market_data:
            return {}
            
        return {
            'symbol': symbol,
            'timestamp': market_data.timestamp.isoformat(),
            'open': market_data.open,
            'high': market_data.high,
            'low': market_data.low,
            'close': market_data.close,
            'volume': market_data.volume,
            'adjusted_close': market_data.adjusted_close
        }
        
    async def _get_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        """Get fundamental data"""
        provider = self.data_providers.get('yahoo')
        if not provider:
            return {}
            
        async with provider:
            fundamental_data = await provider.get_fundamental_data(symbol)
            
        if not fundamental_data:
            return {}
            
        return {
            'symbol': symbol,
            'quarter': fundamental_data.quarter,
            'revenue': fundamental_data.revenue,
            'net_income': fundamental_data.net_income,
            'eps': fundamental_data.eps,
            'pe_ratio': fundamental_data.pe_ratio,
            'pb_ratio': fundamental_data.pb_ratio,
            'debt_to_equity': fundamental_data.debt_to_equity,
            'roe': fundamental_data.roe,
            'roa': fundamental_data.roa,
            'free_cash_flow': fundamental_data.free_cash_flow
        }
        
    async def _handle_status_request(self, message: AgentMessage) -> AgentResponse:
        """Handle status request"""
        return AgentResponse(
            success=True,
            data={
                'cache_size': len(self.cache),
                'providers': list(self.data_providers.keys()),
                'status': 'healthy'
            },
            confidence=1.0
        )