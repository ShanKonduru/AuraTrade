import asyncio
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
import requests
from bs4 import BeautifulSoup
from loguru import logger

from ..base_agent import BaseAgent, AgentMessage, AgentResponse
from ...agent_types import AgentType, MessageType, ActionType, SignalConfidence


@dataclass
class FinancialMetrics:
    """Financial metrics structure"""
    symbol: str
    period: str
    revenue: Optional[float] = None
    net_income: Optional[float] = None
    total_assets: Optional[float] = None
    total_liabilities: Optional[float] = None
    shareholders_equity: Optional[float] = None
    cash_and_equivalents: Optional[float] = None
    free_cash_flow: Optional[float] = None
    operating_cash_flow: Optional[float] = None
    total_debt: Optional[float] = None
    
    # Calculated ratios
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    debt_to_equity: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None


@dataclass
class ValuationModel:
    """Valuation model result"""
    model_name: str
    intrinsic_value: float
    current_price: float
    upside_potential: float
    confidence: float
    assumptions: Dict[str, Any]


@dataclass
class QualitativeAnalysis:
    """Qualitative analysis from document parsing"""
    sentiment_score: float
    key_insights: List[str]
    risk_factors: List[str]
    growth_drivers: List[str]
    management_quality: Optional[float] = None
    competitive_position: Optional[str] = None


class DCFCalculator:
    """Discounted Cash Flow valuation calculator"""
    
    @staticmethod
    def calculate_dcf(
        free_cash_flows: List[float],
        discount_rate: float = 0.10,
        terminal_growth_rate: float = 0.02,
        projection_years: int = 5
    ) -> Dict[str, Any]:
        """Calculate DCF valuation"""
        
        if not free_cash_flows or len(free_cash_flows) < 2:
            return {}
            
        # Calculate average growth rate from historical FCF
        growth_rates = []
        for i in range(1, len(free_cash_flows)):
            if free_cash_flows[i-1] != 0:
                growth_rate = (free_cash_flows[i] - free_cash_flows[i-1]) / abs(free_cash_flows[i-1])
                growth_rates.append(growth_rate)
                
        avg_growth_rate = np.mean(growth_rates) if growth_rates else 0.05
        
        # Project future cash flows
        last_fcf = free_cash_flows[-1]
        projected_fcf = []
        
        for year in range(1, projection_years + 1):
            # Gradually decline growth rate to terminal rate
            year_growth_rate = avg_growth_rate * (1 - year / projection_years) + terminal_growth_rate * (year / projection_years)
            projected_fcf.append(last_fcf * ((1 + year_growth_rate) ** year))
            
        # Calculate terminal value
        terminal_fcf = projected_fcf[-1] * (1 + terminal_growth_rate)
        terminal_value = terminal_fcf / (discount_rate - terminal_growth_rate)
        
        # Discount all cash flows to present value
        pv_fcf = []
        for i, fcf in enumerate(projected_fcf):
            pv_fcf.append(fcf / ((1 + discount_rate) ** (i + 1)))
            
        pv_terminal_value = terminal_value / ((1 + discount_rate) ** projection_years)
        
        enterprise_value = sum(pv_fcf) + pv_terminal_value
        
        return {
            'enterprise_value': enterprise_value,
            'projected_fcf': projected_fcf,
            'terminal_value': terminal_value,
            'pv_terminal_value': pv_terminal_value,
            'discount_rate': discount_rate,
            'terminal_growth_rate': terminal_growth_rate,
            'assumptions': {
                'avg_historical_growth': avg_growth_rate,
                'projection_years': projection_years
            }
        }


class PEGCalculator:
    """Price/Earnings to Growth ratio calculator"""
    
    @staticmethod
    def calculate_peg_ratio(pe_ratio: float, growth_rate: float) -> Optional[float]:
        """Calculate PEG ratio"""
        if growth_rate == 0:
            return None
        return pe_ratio / (growth_rate * 100)
    
    @staticmethod
    def calculate_fair_value_peg(
        current_eps: float,
        growth_rate: float,
        target_peg: float = 1.0
    ) -> float:
        """Calculate fair value using PEG method"""
        projected_eps = current_eps * (1 + growth_rate)
        fair_pe = target_peg * (growth_rate * 100)
        return projected_eps * fair_pe


class DocumentAnalyzer:
    """Analyze financial documents using LLM"""
    
    def __init__(self, openai_api_key: str):
        self.llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.vector_store = None
        
    async def analyze_document(self, document_text: str, analysis_type: str) -> QualitativeAnalysis:
        """Analyze document using RAG"""
        try:
            # Split document into chunks
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_text(document_text)
            
            # Create vector store
            self.vector_store = FAISS.from_texts(chunks, self.embeddings)
            
            # Create QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever()
            )
            
            # Analyze different aspects
            insights = await self._extract_insights(qa_chain)
            risk_factors = await self._extract_risk_factors(qa_chain)
            growth_drivers = await self._extract_growth_drivers(qa_chain)
            sentiment = await self._analyze_sentiment(qa_chain)
            
            return QualitativeAnalysis(
                sentiment_score=sentiment,
                key_insights=insights,
                risk_factors=risk_factors,
                growth_drivers=growth_drivers
            )
            
        except Exception as e:
            logger.error(f"Error analyzing document: {e}")
            return QualitativeAnalysis(
                sentiment_score=0.5,
                key_insights=[],
                risk_factors=[],
                growth_drivers=[]
            )
            
    async def _extract_insights(self, qa_chain) -> List[str]:
        """Extract key insights from document"""
        query = "What are the key financial insights and highlights from this document?"
        try:
            result = qa_chain.run(query)
            # Parse result into list of insights
            insights = [insight.strip() for insight in result.split('\n') if insight.strip()]
            return insights[:5]  # Top 5 insights
        except:
            return []
            
    async def _extract_risk_factors(self, qa_chain) -> List[str]:
        """Extract risk factors from document"""
        query = "What are the main risk factors and challenges mentioned in this document?"
        try:
            result = qa_chain.run(query)
            risks = [risk.strip() for risk in result.split('\n') if risk.strip()]
            return risks[:5]  # Top 5 risks
        except:
            return []
            
    async def _extract_growth_drivers(self, qa_chain) -> List[str]:
        """Extract growth drivers from document"""
        query = "What are the key growth drivers and opportunities mentioned in this document?"
        try:
            result = qa_chain.run(query)
            drivers = [driver.strip() for driver in result.split('\n') if driver.strip()]
            return drivers[:5]  # Top 5 drivers
        except:
            return []
            
    async def _analyze_sentiment(self, qa_chain) -> float:
        """Analyze overall sentiment"""
        query = "What is the overall sentiment and outlook expressed in this document? Rate from 0 (very negative) to 1 (very positive)."
        try:
            result = qa_chain.run(query)
            # Extract numerical sentiment score
            # This is simplified - in production, use proper sentiment analysis
            if "positive" in result.lower():
                return 0.7
            elif "negative" in result.lower():
                return 0.3
            else:
                return 0.5
        except:
            return 0.5


class FundamentalAnalysisAgent(BaseAgent):
    """Agent for fundamental analysis and intrinsic valuation"""
    
    def __init__(self, message_bus, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.FUNDAMENTAL_ANALYSIS, message_bus, config)
        
        self.dcf_calculator = DCFCalculator()
        self.peg_calculator = PEGCalculator()
        self.document_analyzer = None
        
        # Configuration
        self.discount_rate = config.get('discount_rate', 0.10)
        self.terminal_growth_rate = config.get('terminal_growth_rate', 0.02)
        self.openai_api_key = config.get('openai_api_key')
        
    async def _initialize(self):
        """Initialize the fundamental analysis agent"""
        if self.openai_api_key:
            self.document_analyzer = DocumentAnalyzer(self.openai_api_key)
        logger.info("Fundamental Analysis Agent initialized")
        
    async def _cleanup(self):
        """Cleanup resources"""
        pass
        
    async def process_message(self, message: AgentMessage) -> Optional[AgentResponse]:
        """Process fundamental analysis requests"""
        
        if message.message_type == MessageType.ANALYSIS_REQUEST:
            return await self._handle_analysis_request(message)
            
        return None
        
    async def _handle_analysis_request(self, message: AgentMessage) -> AgentResponse:
        """Handle fundamental analysis requests"""
        try:
            analysis_type = message.payload.get('analysis_type')
            symbol = message.payload.get('symbol')
            
            if not symbol:
                return AgentResponse(
                    success=False,
                    error="Symbol is required for fundamental analysis",
                    confidence=0.0
                )
                
            # Get fundamental data from Data Ingestion Agent
            data_response = await self.request_response(
                recipient=AgentType.DATA_INGESTION,
                message_type=MessageType.DATA_REQUEST,
                payload={
                    'request_type': 'fundamental',
                    'symbol': symbol
                }
            )
            
            if not data_response or not data_response.success:
                return AgentResponse(
                    success=False,
                    error="Failed to get fundamental data",
                    confidence=0.0
                )
                
            # Perform fundamental analysis
            analysis_result = await self._perform_fundamental_analysis(
                data_response.data, symbol
            )
            
            return AgentResponse(
                success=True,
                data=analysis_result,
                confidence=analysis_result.get('overall_confidence', 0.5),
                metadata={'symbol': symbol, 'analysis_type': analysis_type}
            )
            
        except Exception as e:
            logger.error(f"Error in fundamental analysis: {e}")
            return AgentResponse(
                success=False,
                error=str(e),
                confidence=0.0
            )
            
    async def _perform_fundamental_analysis(self, data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Perform comprehensive fundamental analysis"""
        
        # Create financial metrics object
        metrics = self._create_financial_metrics(data)
        
        # Calculate financial ratios
        calculated_ratios = await self._calculate_financial_ratios(metrics)
        
        # Perform valuation analysis
        valuation_models = await self._perform_valuation_analysis(metrics, symbol)
        
        # Analyze financial health
        health_score = await self._analyze_financial_health(metrics)
        
        # Generate recommendation
        recommendation = await self._generate_fundamental_recommendation(
            valuation_models, health_score, calculated_ratios
        )
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'financial_metrics': asdict(metrics),
            'calculated_ratios': calculated_ratios,
            'valuation_models': [asdict(model) for model in valuation_models],
            'financial_health_score': health_score,
            'recommendation': recommendation,
            'overall_confidence': recommendation.get('confidence', 0.5)
        }
        
    def _create_financial_metrics(self, data: Dict[str, Any]) -> FinancialMetrics:
        """Create financial metrics object from data"""
        return FinancialMetrics(
            symbol=data.get('symbol', ''),
            period='current',
            revenue=data.get('revenue'),
            net_income=data.get('net_income'),
            pe_ratio=data.get('pe_ratio'),
            pb_ratio=data.get('pb_ratio'),
            debt_to_equity=data.get('debt_to_equity'),
            roe=data.get('roe'),
            roa=data.get('roa'),
            free_cash_flow=data.get('free_cash_flow')
        )
        
    async def _calculate_financial_ratios(self, metrics: FinancialMetrics) -> Dict[str, Any]:
        """Calculate additional financial ratios"""
        ratios = {}
        
        try:
            # Profitability ratios
            if metrics.revenue and metrics.net_income:
                ratios['net_margin'] = (metrics.net_income / metrics.revenue) * 100
                
            # Efficiency ratios
            if metrics.total_assets and metrics.revenue:
                ratios['asset_turnover'] = metrics.revenue / metrics.total_assets
                
            # Liquidity ratios (would need more detailed balance sheet data)
            # This is simplified for demonstration
            
            # Growth potential indicators
            if metrics.pe_ratio and metrics.roe:
                ratios['peg_estimate'] = metrics.pe_ratio / (metrics.roe * 100) if metrics.roe > 0 else None
                
        except Exception as e:
            logger.error(f"Error calculating ratios: {e}")
            
        return ratios
        
    async def _perform_valuation_analysis(self, metrics: FinancialMetrics, symbol: str) -> List[ValuationModel]:
        """Perform multiple valuation methods"""
        models = []
        
        try:
            # Get current market price (simplified)
            current_price = 100.0  # This would come from real-time data
            
            # P/E based valuation
            if metrics.pe_ratio:
                industry_pe = 15.0  # This would come from industry data
                fair_value_pe = current_price * (industry_pe / metrics.pe_ratio) if metrics.pe_ratio > 0 else current_price
                
                models.append(ValuationModel(
                    model_name="P/E Valuation",
                    intrinsic_value=fair_value_pe,
                    current_price=current_price,
                    upside_potential=(fair_value_pe - current_price) / current_price,
                    confidence=0.6,
                    assumptions={'industry_pe': industry_pe, 'current_pe': metrics.pe_ratio}
                ))
                
            # P/B based valuation
            if metrics.pb_ratio:
                industry_pb = 2.0  # This would come from industry data
                fair_value_pb = current_price * (industry_pb / metrics.pb_ratio) if metrics.pb_ratio > 0 else current_price
                
                models.append(ValuationModel(
                    model_name="P/B Valuation",
                    intrinsic_value=fair_value_pb,
                    current_price=current_price,
                    upside_potential=(fair_value_pb - current_price) / current_price,
                    confidence=0.5,
                    assumptions={'industry_pb': industry_pb, 'current_pb': metrics.pb_ratio}
                ))
                
            # DCF valuation (simplified)
            if metrics.free_cash_flow:
                # This is highly simplified - would need historical FCF data
                historical_fcf = [metrics.free_cash_flow * 0.8, metrics.free_cash_flow * 0.9, metrics.free_cash_flow]
                dcf_result = self.dcf_calculator.calculate_dcf(
                    historical_fcf, 
                    self.discount_rate, 
                    self.terminal_growth_rate
                )
                
                if dcf_result:
                    # Assuming 1 billion shares (would need actual share count)
                    shares_outstanding = 1_000_000_000
                    intrinsic_value_per_share = dcf_result['enterprise_value'] / shares_outstanding
                    
                    models.append(ValuationModel(
                        model_name="DCF Valuation",
                        intrinsic_value=intrinsic_value_per_share,
                        current_price=current_price,
                        upside_potential=(intrinsic_value_per_share - current_price) / current_price,
                        confidence=0.7,
                        assumptions=dcf_result['assumptions']
                    ))
                    
        except Exception as e:
            logger.error(f"Error in valuation analysis: {e}")
            
        return models
        
    async def _analyze_financial_health(self, metrics: FinancialMetrics) -> Dict[str, Any]:
        """Analyze overall financial health"""
        health_factors = []
        
        try:
            # Profitability health
            if metrics.roe and metrics.roe > 0.15:
                health_factors.append({'factor': 'Strong ROE', 'score': 0.8, 'weight': 0.3})
            elif metrics.roe and metrics.roe > 0.10:
                health_factors.append({'factor': 'Adequate ROE', 'score': 0.6, 'weight': 0.3})
            else:
                health_factors.append({'factor': 'Weak ROE', 'score': 0.3, 'weight': 0.3})
                
            # Debt health
            if metrics.debt_to_equity and metrics.debt_to_equity < 0.3:
                health_factors.append({'factor': 'Low Debt', 'score': 0.9, 'weight': 0.2})
            elif metrics.debt_to_equity and metrics.debt_to_equity < 0.6:
                health_factors.append({'factor': 'Moderate Debt', 'score': 0.6, 'weight': 0.2})
            else:
                health_factors.append({'factor': 'High Debt', 'score': 0.2, 'weight': 0.2})
                
            # Valuation health
            if metrics.pe_ratio and metrics.pe_ratio < 15:
                health_factors.append({'factor': 'Undervalued P/E', 'score': 0.8, 'weight': 0.2})
            elif metrics.pe_ratio and metrics.pe_ratio < 25:
                health_factors.append({'factor': 'Fair P/E', 'score': 0.6, 'weight': 0.2})
            else:
                health_factors.append({'factor': 'Expensive P/E', 'score': 0.3, 'weight': 0.2})
                
            # Calculate weighted health score
            total_score = sum(factor['score'] * factor['weight'] for factor in health_factors)
            total_weight = sum(factor['weight'] for factor in health_factors)
            health_score = total_score / total_weight if total_weight > 0 else 0.5
            
            return {
                'overall_score': health_score,
                'factors': health_factors,
                'health_grade': self._get_health_grade(health_score)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing financial health: {e}")
            return {'overall_score': 0.5, 'factors': [], 'health_grade': 'C'}
            
    def _get_health_grade(self, score: float) -> str:
        """Convert health score to letter grade"""
        if score >= 0.8:
            return 'A'
        elif score >= 0.7:
            return 'B'
        elif score >= 0.6:
            return 'C'
        elif score >= 0.4:
            return 'D'
        else:
            return 'F'
            
    async def _generate_fundamental_recommendation(self, 
                                                 valuation_models: List[ValuationModel],
                                                 health_score: Dict[str, Any],
                                                 ratios: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fundamental analysis recommendation"""
        
        if not valuation_models:
            return {
                'action': ActionType.HOLD.value,
                'confidence': 0.0,
                'reasoning': "Insufficient data for valuation"
            }
            
        # Calculate average upside potential
        upside_potentials = [model.upside_potential for model in valuation_models]
        avg_upside = np.mean(upside_potentials)
        
        # Weight by model confidence
        weighted_upside = sum(model.upside_potential * model.confidence for model in valuation_models)
        total_confidence = sum(model.confidence for model in valuation_models)
        final_upside = weighted_upside / total_confidence if total_confidence > 0 else avg_upside
        
        # Incorporate health score
        health_adjustment = health_score['overall_score']
        adjusted_confidence = (total_confidence / len(valuation_models)) * health_adjustment
        
        # Generate recommendation
        if final_upside > 0.20 and health_adjustment > 0.6:  # 20% upside with good health
            action = ActionType.ACCUMULATE.value
            confidence = min(adjusted_confidence * 1.2, 1.0)
        elif final_upside > 0.10 and health_adjustment > 0.5:  # 10% upside with fair health
            action = ActionType.BUY.value
            confidence = adjusted_confidence
        elif final_upside < -0.15 or health_adjustment < 0.4:  # 15% downside or poor health
            action = ActionType.DIVEST.value
            confidence = adjusted_confidence
        else:
            action = ActionType.HOLD.value
            confidence = adjusted_confidence * 0.8
            
        return {
            'action': action,
            'confidence': confidence,
            'target_upside': final_upside,
            'health_grade': health_score['health_grade'],
            'valuation_consensus': len([m for m in valuation_models if m.upside_potential > 0]),
            'reasoning': f"Fundamental analysis shows {final_upside:.1%} upside potential with {health_score['health_grade']} financial health"
        }