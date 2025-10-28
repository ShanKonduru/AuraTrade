import numpy as np
import pandas as pd
import talib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import asyncio
from loguru import logger

from ..base_agent import BaseAgent, AgentMessage, AgentResponse
from ...agent_types import AgentType, MessageType, ActionType, SignalConfidence


@dataclass
class TechnicalSignal:
    """Technical analysis signal structure"""
    indicator: str
    value: float
    signal: ActionType
    confidence: SignalConfidence
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class PatternRecognition:
    """Chart pattern recognition result"""
    pattern_name: str
    confidence: float
    start_date: datetime
    end_date: datetime
    target_price: Optional[float]
    stop_loss: Optional[float]


class TechnicalIndicators:
    """Technical indicators calculator"""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        return talib.RSI(prices.values, timeperiod=period)
    
    @staticmethod
    def calculate_macd(prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        macd, signal, histogram = talib.MACD(prices.values)
        return macd, signal, histogram
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        upper, middle, lower = talib.BBANDS(prices.values, timeperiod=period, nbdevup=std, nbdevdn=std)
        return upper, middle, lower
    
    @staticmethod
    def calculate_moving_averages(prices: pd.Series, periods: List[int]) -> Dict[int, pd.Series]:
        """Calculate multiple moving averages"""
        mas = {}
        for period in periods:
            mas[period] = talib.SMA(prices.values, timeperiod=period)
        return mas
    
    @staticmethod
    def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        slowk, slowd = talib.STOCH(high.values, low.values, close.values)
        return slowk, slowd
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        return talib.ATR(high.values, low.values, close.values, timeperiod=period)
    
    @staticmethod
    def calculate_volume_profile(prices: pd.Series, volumes: pd.Series, bins: int = 50) -> Dict[str, Any]:
        """Calculate Volume Profile"""
        price_min, price_max = prices.min(), prices.max()
        price_levels = np.linspace(price_min, price_max, bins)
        
        volume_profile = {}
        for i in range(len(price_levels) - 1):
            mask = (prices >= price_levels[i]) & (prices < price_levels[i + 1])
            volume_profile[f"{price_levels[i]:.2f}-{price_levels[i+1]:.2f}"] = volumes[mask].sum()
            
        # Find Point of Control (POC) - price level with highest volume
        poc_level = max(volume_profile.items(), key=lambda x: x[1])
        
        return {
            'profile': volume_profile,
            'poc': poc_level[0],
            'poc_volume': poc_level[1]
        }


class PatternDetector:
    """Chart pattern detection using technical analysis"""
    
    @staticmethod
    def detect_head_and_shoulders(highs: pd.Series, lows: pd.Series) -> Optional[PatternRecognition]:
        """Detect Head and Shoulders pattern"""
        try:
            # Simplified H&S detection - look for three peaks
            peaks = []
            for i in range(1, len(highs) - 1):
                if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i+1]:
                    peaks.append((i, highs.iloc[i]))
            
            if len(peaks) >= 3:
                # Check if middle peak is highest (head)
                head_idx = max(peaks, key=lambda x: x[1])
                left_shoulder = min([p for p in peaks if p[0] < head_idx[0]], key=lambda x: x[0], default=None)
                right_shoulder = min([p for p in peaks if p[0] > head_idx[0]], key=lambda x: x[0], default=None)
                
                if left_shoulder and right_shoulder:
                    # Basic H&S validation
                    if (abs(left_shoulder[1] - right_shoulder[1]) / left_shoulder[1] < 0.05 and  # Similar heights
                        head_idx[1] > left_shoulder[1] * 1.05):  # Head is higher
                        
                        return PatternRecognition(
                            pattern_name="Head and Shoulders",
                            confidence=0.7,
                            start_date=highs.index[left_shoulder[0]],
                            end_date=highs.index[right_shoulder[0]],
                            target_price=min(left_shoulder[1], right_shoulder[1]) * 0.95,
                            stop_loss=head_idx[1] * 1.02
                        )
        except Exception as e:
            logger.error(f"Error detecting Head and Shoulders: {e}")
            
        return None
    
    @staticmethod
    def detect_double_top(highs: pd.Series) -> Optional[PatternRecognition]:
        """Detect Double Top pattern"""
        try:
            peaks = []
            for i in range(1, len(highs) - 1):
                if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i+1]:
                    peaks.append((i, highs.iloc[i]))
            
            if len(peaks) >= 2:
                # Find two similar peaks
                for i in range(len(peaks) - 1):
                    for j in range(i + 1, len(peaks)):
                        peak1, peak2 = peaks[i], peaks[j]
                        height_diff = abs(peak1[1] - peak2[1]) / peak1[1]
                        
                        if height_diff < 0.03:  # Within 3% of each other
                            return PatternRecognition(
                                pattern_name="Double Top",
                                confidence=0.6,
                                start_date=highs.index[peak1[0]],
                                end_date=highs.index[peak2[0]],
                                target_price=min(peak1[1], peak2[1]) * 0.9,
                                stop_loss=max(peak1[1], peak2[1]) * 1.02
                            )
        except Exception as e:
            logger.error(f"Error detecting Double Top: {e}")
            
        return None
    
    @staticmethod
    def detect_support_resistance(prices: pd.Series, window: int = 20) -> Dict[str, List[float]]:
        """Detect support and resistance levels"""
        try:
            # Rolling min/max for support/resistance
            support_levels = prices.rolling(window=window).min().dropna().unique()
            resistance_levels = prices.rolling(window=window).max().dropna().unique()
            
            # Filter levels that appear multiple times
            current_price = prices.iloc[-1]
            
            support = [level for level in support_levels 
                      if level < current_price and 
                      sum(abs(prices - level) < current_price * 0.01) >= 2]
            
            resistance = [level for level in resistance_levels 
                         if level > current_price and 
                         sum(abs(prices - level) < current_price * 0.01) >= 2]
            
            return {
                'support': sorted(support, reverse=True)[:3],  # Top 3 support levels
                'resistance': sorted(resistance)[:3]  # Top 3 resistance levels
            }
            
        except Exception as e:
            logger.error(f"Error detecting support/resistance: {e}")
            return {'support': [], 'resistance': []}


class TechnicalAnalysisAgent(BaseAgent):
    """Agent for technical analysis and pattern recognition"""
    
    def __init__(self, message_bus, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.TECHNICAL_ANALYSIS, message_bus, config)
        
        self.indicators = TechnicalIndicators()
        self.pattern_detector = PatternDetector()
        self.model = None
        self.scaler = MinMaxScaler()
        
        # Configuration
        self.rsi_period = config.get('rsi_period', 14)
        self.ma_periods = config.get('ma_periods', [5, 10, 20, 50, 200])
        self.bollinger_period = config.get('bollinger_period', 20)
        
    async def _initialize(self):
        """Initialize the technical analysis agent"""
        # Initialize ML model for pattern recognition
        await self._initialize_ml_model()
        logger.info("Technical Analysis Agent initialized")
        
    async def _cleanup(self):
        """Cleanup resources"""
        pass
        
    async def _initialize_ml_model(self):
        """Initialize machine learning model for signal prediction"""
        # Simple Random Forest for demonstration
        # In production, this would be a more sophisticated model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
    async def process_message(self, message: AgentMessage) -> Optional[AgentResponse]:
        """Process technical analysis requests"""
        
        if message.message_type == MessageType.ANALYSIS_REQUEST:
            return await self._handle_analysis_request(message)
        elif message.message_type == MessageType.DATA_RESPONSE:
            return await self._handle_data_response(message)
            
        return None
        
    async def _handle_analysis_request(self, message: AgentMessage) -> AgentResponse:
        """Handle technical analysis requests"""
        try:
            analysis_type = message.payload.get('analysis_type')
            symbol = message.payload.get('symbol')
            
            if not symbol:
                return AgentResponse(
                    success=False,
                    error="Symbol is required for technical analysis",
                    confidence=0.0
                )
                
            # Request historical data from Data Ingestion Agent
            data_response = await self.request_response(
                recipient=AgentType.DATA_INGESTION,
                message_type=MessageType.DATA_REQUEST,
                payload={
                    'request_type': 'historical',
                    'symbol': symbol,
                    'period': '1y'
                }
            )
            
            if not data_response or not data_response.success:
                return AgentResponse(
                    success=False,
                    error="Failed to get historical data",
                    confidence=0.0
                )
                
            # Perform technical analysis
            analysis_result = await self._perform_technical_analysis(
                data_response.data, analysis_type
            )
            
            return AgentResponse(
                success=True,
                data=analysis_result,
                confidence=analysis_result.get('overall_confidence', 0.5),
                metadata={'symbol': symbol, 'analysis_type': analysis_type}
            )
            
        except Exception as e:
            logger.error(f"Error in technical analysis: {e}")
            return AgentResponse(
                success=False,
                error=str(e),
                confidence=0.0
            )
            
    async def _handle_data_response(self, message: AgentMessage) -> Optional[AgentResponse]:
        """Handle data responses for ongoing analysis"""
        # This would handle real-time data updates
        return None
        
    async def _perform_technical_analysis(self, data: Dict[str, Any], analysis_type: str) -> Dict[str, Any]:
        """Perform comprehensive technical analysis"""
        
        # Convert data to DataFrame
        df = pd.DataFrame(data['data'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Calculate all indicators
        indicators = await self._calculate_all_indicators(df)
        
        # Detect patterns
        patterns = await self._detect_patterns(df)
        
        # Generate signals
        signals = await self._generate_signals(df, indicators, patterns)
        
        # Calculate overall recommendation
        recommendation = await self._calculate_recommendation(signals)
        
        return {
            'symbol': data['symbol'],
            'timestamp': datetime.now().isoformat(),
            'indicators': indicators,
            'patterns': patterns,
            'signals': signals,
            'recommendation': recommendation,
            'overall_confidence': recommendation['confidence']
        }
        
    async def _calculate_all_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate all technical indicators"""
        indicators = {}
        
        try:
            # RSI
            rsi = self.indicators.calculate_rsi(df['close'], self.rsi_period)
            indicators['rsi'] = {
                'current': float(rsi[-1]) if not pd.isna(rsi[-1]) else None,
                'signal': 'oversold' if rsi[-1] < 30 else 'overbought' if rsi[-1] > 70 else 'neutral'
            }
            
            # MACD
            macd, signal, histogram = self.indicators.calculate_macd(df['close'])
            indicators['macd'] = {
                'macd': float(macd[-1]) if not pd.isna(macd[-1]) else None,
                'signal': float(signal[-1]) if not pd.isna(signal[-1]) else None,
                'histogram': float(histogram[-1]) if not pd.isna(histogram[-1]) else None,
                'bullish_crossover': macd[-1] > signal[-1] and macd[-2] <= signal[-2] if len(macd) > 1 else False
            }
            
            # Bollinger Bands
            upper, middle, lower = self.indicators.calculate_bollinger_bands(df['close'])
            current_price = df['close'].iloc[-1]
            indicators['bollinger_bands'] = {
                'upper': float(upper[-1]) if not pd.isna(upper[-1]) else None,
                'middle': float(middle[-1]) if not pd.isna(middle[-1]) else None,
                'lower': float(lower[-1]) if not pd.isna(lower[-1]) else None,
                'position': 'above_upper' if current_price > upper[-1] else 'below_lower' if current_price < lower[-1] else 'middle'
            }
            
            # Moving Averages
            mas = self.indicators.calculate_moving_averages(df['close'], self.ma_periods)
            indicators['moving_averages'] = {}
            for period, ma in mas.items():
                if not pd.isna(ma[-1]):
                    indicators['moving_averages'][f'ma_{period}'] = {
                        'value': float(ma[-1]),
                        'trend': 'bullish' if current_price > ma[-1] else 'bearish'
                    }
            
            # Stochastic
            slowk, slowd = self.indicators.calculate_stochastic(df['high'], df['low'], df['close'])
            indicators['stochastic'] = {
                'k': float(slowk[-1]) if not pd.isna(slowk[-1]) else None,
                'd': float(slowd[-1]) if not pd.isna(slowd[-1]) else None,
                'signal': 'oversold' if slowk[-1] < 20 else 'overbought' if slowk[-1] > 80 else 'neutral'
            }
            
            # ATR
            atr = self.indicators.calculate_atr(df['high'], df['low'], df['close'])
            indicators['atr'] = {
                'current': float(atr[-1]) if not pd.isna(atr[-1]) else None,
                'volatility': 'high' if atr[-1] > atr[:-1].mean() * 1.5 else 'low' if atr[-1] < atr[:-1].mean() * 0.5 else 'normal'
            }
            
            # Volume Profile
            volume_profile = self.indicators.calculate_volume_profile(df['close'], df['volume'])
            indicators['volume_profile'] = volume_profile
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            
        return indicators
        
    async def _detect_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect chart patterns"""
        patterns = []
        
        try:
            # Head and Shoulders
            h_and_s = self.pattern_detector.detect_head_and_shoulders(df['high'], df['low'])
            if h_and_s:
                patterns.append({
                    'pattern': h_and_s.pattern_name,
                    'confidence': h_and_s.confidence,
                    'signal': 'bearish',
                    'target_price': h_and_s.target_price,
                    'stop_loss': h_and_s.stop_loss
                })
                
            # Double Top
            double_top = self.pattern_detector.detect_double_top(df['high'])
            if double_top:
                patterns.append({
                    'pattern': double_top.pattern_name,
                    'confidence': double_top.confidence,
                    'signal': 'bearish',
                    'target_price': double_top.target_price,
                    'stop_loss': double_top.stop_loss
                })
                
            # Support/Resistance
            support_resistance = self.pattern_detector.detect_support_resistance(df['close'])
            patterns.append({
                'pattern': 'Support/Resistance',
                'support_levels': support_resistance['support'],
                'resistance_levels': support_resistance['resistance']
            })
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
            
        return patterns
        
    async def _generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate trading signals based on indicators and patterns"""
        signals = []
        
        try:
            current_price = df['close'].iloc[-1]
            
            # RSI Signal
            if indicators.get('rsi', {}).get('current'):
                rsi_value = indicators['rsi']['current']
                if rsi_value < 30:
                    signals.append({
                        'indicator': 'RSI',
                        'signal': ActionType.BUY.value,
                        'confidence': 0.7,
                        'reasoning': f"RSI oversold at {rsi_value:.1f}"
                    })
                elif rsi_value > 70:
                    signals.append({
                        'indicator': 'RSI',
                        'signal': ActionType.SELL.value,
                        'confidence': 0.7,
                        'reasoning': f"RSI overbought at {rsi_value:.1f}"
                    })
                    
            # MACD Signal
            macd_data = indicators.get('macd', {})
            if macd_data.get('bullish_crossover'):
                signals.append({
                    'indicator': 'MACD',
                    'signal': ActionType.BUY.value,
                    'confidence': 0.6,
                    'reasoning': "MACD bullish crossover"
                })
                
            # Moving Average Trend
            ma_data = indicators.get('moving_averages', {})
            bullish_mas = sum(1 for ma in ma_data.values() if ma['trend'] == 'bullish')
            bearish_mas = sum(1 for ma in ma_data.values() if ma['trend'] == 'bearish')
            
            if bullish_mas > bearish_mas * 1.5:
                signals.append({
                    'indicator': 'Moving Averages',
                    'signal': ActionType.BUY.value,
                    'confidence': 0.5,
                    'reasoning': f"{bullish_mas} bullish vs {bearish_mas} bearish MAs"
                })
            elif bearish_mas > bullish_mas * 1.5:
                signals.append({
                    'indicator': 'Moving Averages',
                    'signal': ActionType.SELL.value,
                    'confidence': 0.5,
                    'reasoning': f"{bearish_mas} bearish vs {bullish_mas} bullish MAs"
                })
                
            # Pattern Signals
            for pattern in patterns:
                if pattern.get('signal'):
                    signals.append({
                        'indicator': 'Pattern',
                        'signal': ActionType.SELL.value if pattern['signal'] == 'bearish' else ActionType.BUY.value,
                        'confidence': pattern.get('confidence', 0.5),
                        'reasoning': f"{pattern['pattern']} pattern detected"
                    })
                    
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            
        return signals
        
    async def _calculate_recommendation(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall recommendation from all signals"""
        
        if not signals:
            return {
                'action': ActionType.HOLD.value,
                'confidence': 0.0,
                'reasoning': "No signals available"
            }
            
        # Weight signals by confidence
        buy_weight = sum(s['confidence'] for s in signals if s['signal'] == ActionType.BUY.value)
        sell_weight = sum(s['confidence'] for s in signals if s['signal'] == ActionType.SELL.value)
        total_weight = buy_weight + sell_weight
        
        if total_weight == 0:
            return {
                'action': ActionType.HOLD.value,
                'confidence': 0.0,
                'reasoning': "No actionable signals"
            }
            
        buy_ratio = buy_weight / total_weight
        sell_ratio = sell_weight / total_weight
        
        # Determine action based on weighted signals
        if buy_ratio > 0.6:
            action = ActionType.BUY.value
            confidence = buy_ratio
        elif sell_ratio > 0.6:
            action = ActionType.SELL.value
            confidence = sell_ratio
        else:
            action = ActionType.HOLD.value
            confidence = 1 - abs(buy_ratio - sell_ratio)  # Higher confidence when signals are balanced
            
        return {
            'action': action,
            'confidence': confidence,
            'buy_signals': len([s for s in signals if s['signal'] == ActionType.BUY.value]),
            'sell_signals': len([s for s in signals if s['signal'] == ActionType.SELL.value]),
            'reasoning': f"Technical analysis suggests {action} with {confidence:.1%} confidence"
        }