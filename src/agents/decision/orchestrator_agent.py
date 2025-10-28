import asyncio
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from loguru import logger

from ..base_agent import BaseAgent, AgentMessage, AgentResponse
from ...agent_types import AgentType, MessageType, ActionType, SignalConfidence, RiskLevel


@dataclass
class AgentSignal:
    """Signal from an individual agent"""
    agent_type: AgentType
    action: ActionType
    confidence: float
    reasoning: str
    metadata: Dict[str, Any]
    timestamp: datetime


@dataclass
class TradingDecision:
    """Final trading decision from orchestrator"""
    symbol: str
    action: ActionType
    position_size_pct: float
    confidence: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    reasoning: str
    risk_level: RiskLevel
    expected_holding_period: str
    signals_summary: Dict[str, Any]


@dataclass
class ConflictResolution:
    """Resolution of conflicting signals"""
    conflict_type: str
    conflicting_agents: List[AgentType]
    resolution_method: str
    final_decision: ActionType
    confidence_adjustment: float
    reasoning: str


class ChainOfThoughtProcessor:
    """Chain-of-Thought reasoning for trading decisions"""
    
    def __init__(self, llm_manager):
        self.llm_manager = llm_manager
        
    def _initialize_reasoning_chain(self):
        """Initialize the Chain-of-Thought reasoning chain"""
        
        reasoning_template = """
        You are an expert AI trading orchestrator analyzing multiple agent signals to make trading decisions.
        
        Technical Analysis Signal: {technical_signal}
        Fundamental Analysis Signal: {fundamental_signal}
        Sentiment Analysis Signal: {sentiment_signal}
        
        Current Market Context:
        - Symbol: {symbol}
        - Current Price: {current_price}
        - Market Regime: {market_regime}
        - Volatility Level: {volatility_level}
        
        Please provide a step-by-step Chain-of-Thought analysis:
        
        1. SIGNAL ANALYSIS:
           - What does each agent signal suggest?
           - What are the confidence levels?
           - Are there any conflicts between signals?
        
        2. SIGNAL WEIGHTING:
           - How should each signal be weighted given current market conditions?
           - Which signals are most reliable in this context?
        
        3. CONFLICT RESOLUTION (if applicable):
           - Why are the agents disagreeing?
           - Which signal should take precedence and why?
        
        4. RISK ASSESSMENT:
           - What are the key risks of this trade?
           - How does market volatility affect the decision?
        
        5. POSITION SIZING:
           - What position size is appropriate given the confidence and risk?
           - Should this be a full position or partial?
        
        6. FINAL DECISION:
           - Action: BUY/SELL/HOLD/ACCUMULATE/DIVEST
           - Confidence: 0.0 to 1.0
           - Stop Loss level (if applicable)
           - Take Profit level (if applicable)
           - Expected holding period
        
        Provide your reasoning in a structured format with clear justification for each decision.
        """
        
        self.reasoning_prompt = PromptTemplate(
            input_variables=[
                "technical_signal", "fundamental_signal", "sentiment_signal",
                "symbol", "current_price", "market_regime", "volatility_level"
            ],
            template=reasoning_template
        )
        
        self.reasoning_chain = LLMChain(
            llm=self.llm,
            prompt=self.reasoning_prompt
        )
        
    async def analyze_signals(self, 
                            technical_signal: Dict[str, Any],
                            fundamental_signal: Dict[str, Any],
                            sentiment_signal: Dict[str, Any],
                            market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform Chain-of-Thought analysis of all signals"""
        
        if not self.llm_manager:
            return {
                'reasoning': "LLM not available - using default logic",
                'parsed_decision': self._default_decision(),
                'timestamp': datetime.now().isoformat()
            }
        
        try:
            # Create comprehensive prompt
            prompt = self._create_analysis_prompt(
                technical_signal, fundamental_signal, sentiment_signal, market_context
            )
            
            # Get LLM analysis
            response = await self.llm_manager.generate_text(prompt)
            
            if response.success:
                parsed_reasoning = self._parse_reasoning_result(response.content)
                return {
                    'reasoning': response.content,
                    'parsed_decision': parsed_reasoning,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                logger.error(f"LLM analysis failed: {response.error}")
                return {
                    'reasoning': f"LLM analysis failed: {response.error}",
                    'parsed_decision': self._default_decision(),
                    'timestamp': datetime.now().isoformat()
                }
            
        except Exception as e:
            logger.error(f"Error in Chain-of-Thought analysis: {e}")
            return {
                'reasoning': f"Error in LLM analysis: {str(e)}",
                'parsed_decision': self._default_decision(),
                'timestamp': datetime.now().isoformat()
            }
            
    def _create_analysis_prompt(self, technical_signal, fundamental_signal, sentiment_signal, market_context) -> str:
        """Create comprehensive analysis prompt"""
        return f"""
You are an expert AI trading orchestrator analyzing multiple agent signals to make trading decisions.

Technical Analysis Signal: {json.dumps(technical_signal, indent=2)}
Fundamental Analysis Signal: {json.dumps(fundamental_signal, indent=2)}
Sentiment Analysis Signal: {json.dumps(sentiment_signal, indent=2)}

Current Market Context:
- Symbol: {market_context.get('symbol', 'UNKNOWN')}
- Current Price: {market_context.get('current_price', 0)}
- Market Regime: {market_context.get('market_regime', 'UNKNOWN')}
- Volatility Level: {market_context.get('volatility_level', 'MEDIUM')}

Please provide a step-by-step Chain-of-Thought analysis:

1. SIGNAL ANALYSIS:
   - What does each agent signal suggest?
   - Are the signals aligned or conflicting?
   - Which signals are strongest and why?

2. MARKET CONTEXT:
   - How do market conditions affect the signals?
   - What are the key risks and opportunities?

3. DECISION SYNTHESIS:
   - What is the recommended action? (BUY/SELL/HOLD)
   - What confidence level (0.0 to 1.0)?
   - What position size percentage?
   - What stop loss and take profit levels?

4. RISK ASSESSMENT:
   - What are the main risks?
   - How should position be sized?

Please format your final recommendation as:
ACTION: [BUY/SELL/HOLD]
CONFIDENCE: [0.0-1.0]
POSITION_SIZE: [0.0-1.0]
STOP_LOSS: [price or percentage]
TAKE_PROFIT: [price or percentage]
REASONING: [brief summary]
"""
            
    def _parse_reasoning_result(self, reasoning_text: str) -> Dict[str, Any]:
        """Parse the LLM reasoning output into structured decision"""
        
        # This is a simplified parser - in production, would use more robust parsing
        lines = reasoning_text.split('\n')
        decision = {
            'action': ActionType.HOLD,
            'confidence': 0.5,
            'stop_loss': None,
            'take_profit': None,
            'position_size': 0.1,
            'holding_period': 'medium_term'
        }
        
        try:
            for line in lines:
                line_lower = line.lower().strip()
                
                if 'action:' in line_lower:
                    if 'buy' in line_lower:
                        decision['action'] = ActionType.BUY
                    elif 'sell' in line_lower:
                        decision['action'] = ActionType.SELL
                    elif 'accumulate' in line_lower:
                        decision['action'] = ActionType.ACCUMULATE
                    elif 'divest' in line_lower:
                        decision['action'] = ActionType.DIVEST
                        
                elif 'confidence:' in line_lower:
                    # Extract confidence value
                    import re
                    confidence_match = re.search(r'(\d+\.?\d*)', line)
                    if confidence_match:
                        confidence_val = float(confidence_match.group(1))
                        if confidence_val <= 1.0:
                            decision['confidence'] = confidence_val
                        elif confidence_val <= 100:
                            decision['confidence'] = confidence_val / 100.0
                            
                elif 'stop loss' in line_lower or 'stop-loss' in line_lower:
                    price_match = re.search(r'(\d+\.?\d*)', line)
                    if price_match:
                        decision['stop_loss'] = float(price_match.group(1))
                        
                elif 'take profit' in line_lower or 'take-profit' in line_lower:
                    price_match = re.search(r'(\d+\.?\d*)', line)
                    if price_match:
                        decision['take_profit'] = float(price_match.group(1))
                        
        except Exception as e:
            logger.error(f"Error parsing reasoning result: {e}")
            
        return decision
        
    def _default_decision(self) -> Dict[str, Any]:
        """Return default decision when parsing fails"""
        return {
            'action': ActionType.HOLD,
            'confidence': 0.3,
            'stop_loss': None,
            'take_profit': None,
            'position_size': 0.05,
            'holding_period': 'short_term'
        }


class SignalAggregator:
    """Aggregate and weight signals from different agents"""
    
    def __init__(self, config: Dict[str, Any]):
        # Default weights for different agents
        self.agent_weights = config.get('agent_weights', {
            AgentType.TECHNICAL_ANALYSIS: 0.35,
            AgentType.FUNDAMENTAL_ANALYSIS: 0.40,
            AgentType.SENTIMENT_ANALYSIS: 0.25
        })
        
        # Market regime adjustments
        self.regime_adjustments = config.get('regime_adjustments', {
            'bull': {'technical': 1.0, 'fundamental': 1.2, 'sentiment': 0.8},
            'bear': {'technical': 1.2, 'fundamental': 1.0, 'sentiment': 1.0},
            'sideways': {'technical': 1.3, 'fundamental': 0.9, 'sentiment': 0.8},
            'volatile': {'technical': 0.8, 'fundamental': 1.1, 'sentiment': 1.1}
        })
        
    def aggregate_signals(self, 
                         signals: List[AgentSignal],
                         market_regime: str = 'unknown') -> Dict[str, Any]:
        """Aggregate multiple agent signals into consensus"""
        
        if not signals:
            return {
                'consensus_action': ActionType.HOLD,
                'consensus_confidence': 0.0,
                'signal_strength': 0.0,
                'conflicts': []
            }
            
        # Adjust weights based on market regime
        adjusted_weights = self._adjust_weights_for_regime(market_regime)
        
        # Convert actions to numerical scores for aggregation
        action_scores = {
            ActionType.DIVEST: -2,
            ActionType.SELL: -1,
            ActionType.HOLD: 0,
            ActionType.BUY: 1,
            ActionType.ACCUMULATE: 2
        }
        
        # Calculate weighted consensus
        weighted_score = 0
        total_weight = 0
        confidence_sum = 0
        
        for signal in signals:
            agent_weight = adjusted_weights.get(signal.agent_type, 0.33)
            action_score = action_scores.get(signal.action, 0)
            
            weighted_score += action_score * signal.confidence * agent_weight
            confidence_sum += signal.confidence * agent_weight
            total_weight += agent_weight
            
        if total_weight == 0:
            return {
                'consensus_action': ActionType.HOLD,
                'consensus_confidence': 0.0,
                'signal_strength': 0.0,
                'conflicts': []
            }
            
        # Calculate final consensus
        avg_score = weighted_score / total_weight
        avg_confidence = confidence_sum / total_weight
        
        # Convert score back to action
        consensus_action = self._score_to_action(avg_score)
        
        # Detect conflicts
        conflicts = self._detect_conflicts(signals)
        
        # Adjust confidence based on conflicts
        conflict_penalty = min(len(conflicts) * 0.1, 0.3)
        final_confidence = max(avg_confidence - conflict_penalty, 0.1)
        
        return {
            'consensus_action': consensus_action,
            'consensus_confidence': final_confidence,
            'signal_strength': abs(avg_score),
            'conflicts': conflicts,
            'individual_signals': [asdict(signal) for signal in signals]
        }
        
    def _adjust_weights_for_regime(self, market_regime: str) -> Dict[AgentType, float]:
        """Adjust agent weights based on market regime"""
        base_weights = self.agent_weights.copy()
        
        if market_regime in self.regime_adjustments:
            adjustments = self.regime_adjustments[market_regime]
            
            if AgentType.TECHNICAL_ANALYSIS in base_weights:
                base_weights[AgentType.TECHNICAL_ANALYSIS] *= adjustments.get('technical', 1.0)
            if AgentType.FUNDAMENTAL_ANALYSIS in base_weights:
                base_weights[AgentType.FUNDAMENTAL_ANALYSIS] *= adjustments.get('fundamental', 1.0)
            if AgentType.SENTIMENT_ANALYSIS in base_weights:
                base_weights[AgentType.SENTIMENT_ANALYSIS] *= adjustments.get('sentiment', 1.0)
                
        # Normalize weights
        total_weight = sum(base_weights.values())
        if total_weight > 0:
            for agent_type in base_weights:
                base_weights[agent_type] /= total_weight
                
        return base_weights
        
    def _score_to_action(self, score: float) -> ActionType:
        """Convert numerical score to action"""
        if score >= 1.5:
            return ActionType.ACCUMULATE
        elif score >= 0.3:
            return ActionType.BUY
        elif score <= -1.5:
            return ActionType.DIVEST
        elif score <= -0.3:
            return ActionType.SELL
        else:
            return ActionType.HOLD
            
    def _detect_conflicts(self, signals: List[AgentSignal]) -> List[Dict[str, Any]]:
        """Detect conflicts between agent signals"""
        conflicts = []
        
        # Check for opposing signals
        buy_signals = [s for s in signals if s.action in [ActionType.BUY, ActionType.ACCUMULATE]]
        sell_signals = [s for s in signals if s.action in [ActionType.SELL, ActionType.DIVEST]]
        
        if buy_signals and sell_signals:
            conflicts.append({
                'type': 'buy_vs_sell',
                'buy_agents': [s.agent_type.value for s in buy_signals],
                'sell_agents': [s.agent_type.value for s in sell_signals],
                'severity': 'high'
            })
            
        # Check for high confidence disagreements
        high_conf_signals = [s for s in signals if s.confidence > 0.7]
        if len(high_conf_signals) > 1:
            actions = [s.action for s in high_conf_signals]
            if len(set(actions)) > 1:
                conflicts.append({
                    'type': 'high_confidence_disagreement',
                    'conflicting_signals': [(s.agent_type.value, s.action.value) for s in high_conf_signals],
                    'severity': 'medium'
                })
                
        return conflicts


class OrchestratorAgent(BaseAgent):
    """Central orchestrator agent for trading decisions"""
    
    def __init__(self, message_bus, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.ORCHESTRATOR, message_bus, config)
        
        self.cot_processor = None
        self.signal_aggregator = SignalAggregator(config or {})
        self.pending_analyses = {}
        
        # Configuration
        self.analysis_timeout = config.get('analysis_timeout', 30.0)
        self.min_confidence_threshold = config.get('min_confidence_threshold', 0.3)
        
        # Initialize LLM manager
        self.llm_manager = None
        llm_config = config.get('llm_config', {})
        if llm_config:
            from src.llm import create_llm_manager
            self.llm_manager = create_llm_manager(llm_config.get('providers', {}))
        
    async def _initialize(self):
        """Initialize the orchestrator agent"""
        if self.llm_manager:
            self.cot_processor = ChainOfThoughtProcessor(self.llm_manager)
        logger.info("Orchestrator Agent initialized")
        
    async def _cleanup(self):
        """Cleanup resources"""
        pass
        
    async def process_message(self, message: AgentMessage) -> Optional[AgentResponse]:
        """Process orchestration requests"""
        
        if message.message_type == MessageType.TRADE_SIGNAL:
            return await self._handle_trade_signal_request(message)
        elif message.message_type == MessageType.ANALYSIS_RESPONSE:
            return await self._handle_agent_analysis_response(message)
            
        return None
        
    async def _handle_trade_signal_request(self, message: AgentMessage) -> AgentResponse:
        """Handle requests for trading signals"""
        try:
            symbol = message.payload.get('symbol')
            if not symbol:
                return AgentResponse(
                    success=False,
                    error="Symbol is required for trading signal",
                    confidence=0.0
                )
                
            # Generate comprehensive trading decision
            trading_decision = await self._generate_trading_decision(symbol)
            
            return AgentResponse(
                success=True,
                data=asdict(trading_decision),
                confidence=trading_decision.confidence,
                metadata={'symbol': symbol}
            )
            
        except Exception as e:
            logger.error(f"Error generating trading signal: {e}")
            return AgentResponse(
                success=False,
                error=str(e),
                confidence=0.0
            )
            
    async def _handle_agent_analysis_response(self, message: AgentMessage) -> Optional[AgentResponse]:
        """Handle analysis responses from other agents"""
        # This would be used for real-time signal updates
        return None
        
    async def _generate_trading_decision(self, symbol: str) -> TradingDecision:
        """Generate comprehensive trading decision"""
        
        # Request analysis from all cognitive agents
        analysis_tasks = [
            self._request_technical_analysis(symbol),
            self._request_fundamental_analysis(symbol),
            self._request_sentiment_analysis(symbol)
        ]
        
        # Execute all analyses in parallel
        try:
            technical_result, fundamental_result, sentiment_result = await asyncio.gather(
                *analysis_tasks, return_exceptions=True
            )
        except Exception as e:
            logger.error(f"Error in parallel analysis execution: {e}")
            # Fallback to default decision
            return self._create_default_decision(symbol)
            
        # Convert results to agent signals
        signals = []
        
        if isinstance(technical_result, AgentResponse) and technical_result.success:
            signals.append(self._create_agent_signal(
                AgentType.TECHNICAL_ANALYSIS, technical_result.data
            ))
            
        if isinstance(fundamental_result, AgentResponse) and fundamental_result.success:
            signals.append(self._create_agent_signal(
                AgentType.FUNDAMENTAL_ANALYSIS, fundamental_result.data
            ))
            
        if isinstance(sentiment_result, AgentResponse) and sentiment_result.success:
            signals.append(self._create_agent_signal(
                AgentType.SENTIMENT_ANALYSIS, sentiment_result.data
            ))
            
        if not signals:
            logger.warning("No valid signals received from agents")
            return self._create_default_decision(symbol)
            
        # Aggregate signals
        signal_consensus = self.signal_aggregator.aggregate_signals(signals)
        
        # Perform Chain-of-Thought reasoning if available
        if self.cot_processor:
            market_context = await self._get_market_context(symbol)
            
            cot_analysis = await self.cot_processor.analyze_signals(
                technical_signal=technical_result.data if isinstance(technical_result, AgentResponse) else {},
                fundamental_signal=fundamental_result.data if isinstance(fundamental_result, AgentResponse) else {},
                sentiment_signal=sentiment_result.data if isinstance(sentiment_result, AgentResponse) else {},
                market_context=market_context
            )
            
            # Use LLM decision if available, otherwise use aggregated signals
            if cot_analysis.get('parsed_decision'):
                return self._create_trading_decision_from_cot(
                    symbol, cot_analysis, signal_consensus, signals
                )
                
        # Create decision from aggregated signals
        return self._create_trading_decision_from_consensus(
            symbol, signal_consensus, signals
        )
        
    async def _request_technical_analysis(self, symbol: str) -> AgentResponse:
        """Request technical analysis"""
        return await self.request_response(
            recipient=AgentType.TECHNICAL_ANALYSIS,
            message_type=MessageType.ANALYSIS_REQUEST,
            payload={'analysis_type': 'technical', 'symbol': symbol},
            timeout=self.analysis_timeout
        )
        
    async def _request_fundamental_analysis(self, symbol: str) -> AgentResponse:
        """Request fundamental analysis"""
        return await self.request_response(
            recipient=AgentType.FUNDAMENTAL_ANALYSIS,
            message_type=MessageType.ANALYSIS_REQUEST,
            payload={'analysis_type': 'fundamental', 'symbol': symbol},
            timeout=self.analysis_timeout
        )
        
    async def _request_sentiment_analysis(self, symbol: str) -> AgentResponse:
        """Request sentiment analysis"""
        return await self.request_response(
            recipient=AgentType.SENTIMENT_ANALYSIS,
            message_type=MessageType.ANALYSIS_REQUEST,
            payload={'analysis_type': 'sentiment', 'symbol': symbol},
            timeout=self.analysis_timeout
        )
        
    def _create_agent_signal(self, agent_type: AgentType, analysis_data: Dict[str, Any]) -> AgentSignal:
        """Create agent signal from analysis data"""
        
        recommendation = analysis_data.get('recommendation', {})
        action_str = recommendation.get('action', 'HOLD')
        
        # Convert string to ActionType
        action = ActionType.HOLD
        try:
            action = ActionType(action_str)
        except ValueError:
            logger.warning(f"Unknown action type: {action_str}")
            
        return AgentSignal(
            agent_type=agent_type,
            action=action,
            confidence=recommendation.get('confidence', 0.5),
            reasoning=recommendation.get('reasoning', 'No reasoning provided'),
            metadata=analysis_data,
            timestamp=datetime.now()
        )
        
    async def _get_market_context(self, symbol: str) -> Dict[str, Any]:
        """Get current market context"""
        # This would integrate with real market data
        return {
            'symbol': symbol,
            'current_price': 100.0,  # Would come from real-time data
            'market_regime': 'bull',  # Would be determined by market analysis
            'volatility_level': 'medium'  # Would be calculated from market data
        }
        
    def _create_trading_decision_from_cot(self, 
                                        symbol: str,
                                        cot_analysis: Dict[str, Any],
                                        signal_consensus: Dict[str, Any],
                                        signals: List[AgentSignal]) -> TradingDecision:
        """Create trading decision from Chain-of-Thought analysis"""
        
        parsed_decision = cot_analysis['parsed_decision']
        
        return TradingDecision(
            symbol=symbol,
            action=parsed_decision.get('action', ActionType.HOLD),
            position_size_pct=parsed_decision.get('position_size', 0.1) * 100,
            confidence=parsed_decision.get('confidence', 0.5),
            stop_loss=parsed_decision.get('stop_loss'),
            take_profit=parsed_decision.get('take_profit'),
            reasoning=cot_analysis.get('reasoning', 'Chain-of-Thought analysis'),
            risk_level=self._determine_risk_level(parsed_decision.get('confidence', 0.5)),
            expected_holding_period=parsed_decision.get('holding_period', 'medium_term'),
            signals_summary=signal_consensus
        )
        
    def _create_trading_decision_from_consensus(self, 
                                              symbol: str,
                                              signal_consensus: Dict[str, Any],
                                              signals: List[AgentSignal]) -> TradingDecision:
        """Create trading decision from signal consensus"""
        
        consensus_action = signal_consensus['consensus_action']
        consensus_confidence = signal_consensus['consensus_confidence']
        
        # Calculate position size based on confidence and signal strength
        signal_strength = signal_consensus['signal_strength']
        base_position_size = 0.1  # 10% base position
        
        if consensus_confidence > 0.7 and signal_strength > 1.0:
            position_size = min(base_position_size * 2, 0.25)  # Max 25%
        elif consensus_confidence > 0.5:
            position_size = base_position_size * 1.5
        else:
            position_size = base_position_size * 0.5
            
        # Generate reasoning
        reasoning = f"Consensus from {len(signals)} agents: {consensus_action.value} with {consensus_confidence:.1%} confidence"
        if signal_consensus['conflicts']:
            reasoning += f". Conflicts detected: {len(signal_consensus['conflicts'])}"
            
        return TradingDecision(
            symbol=symbol,
            action=consensus_action,
            position_size_pct=position_size * 100,
            confidence=consensus_confidence,
            stop_loss=None,  # Would be calculated based on volatility
            take_profit=None,  # Would be calculated based on target
            reasoning=reasoning,
            risk_level=self._determine_risk_level(consensus_confidence),
            expected_holding_period='medium_term',
            signals_summary=signal_consensus
        )
        
    def _create_default_decision(self, symbol: str) -> TradingDecision:
        """Create default conservative decision when analysis fails"""
        return TradingDecision(
            symbol=symbol,
            action=ActionType.HOLD,
            position_size_pct=0.0,
            confidence=0.1,
            stop_loss=None,
            take_profit=None,
            reasoning="Insufficient data for trading decision",
            risk_level=RiskLevel.HIGH,
            expected_holding_period='short_term',
            signals_summary={}
        )
        
    def _determine_risk_level(self, confidence: float) -> RiskLevel:
        """Determine risk level based on confidence"""
        if confidence >= 0.8:
            return RiskLevel.LOW
        elif confidence >= 0.6:
            return RiskLevel.MODERATE
        elif confidence >= 0.4:
            return RiskLevel.HIGH
        else:
            return RiskLevel.EXTREME