import asyncio
import aiohttp
import feedparser
import nltk
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import newspaper
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import re
import json
from collections import defaultdict
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from loguru import logger

from ..base_agent import BaseAgent, AgentMessage, AgentResponse
from ...agent_types import AgentType, MessageType, ActionType, SignalConfidence, RiskLevel


@dataclass
class NewsArticle:
    """News article structure"""
    title: str
    url: str
    published_date: datetime
    source: str
    content: Optional[str] = None
    sentiment_score: Optional[float] = None
    relevance_score: Optional[float] = None
    impact_level: Optional[str] = None


@dataclass
class SentimentSignal:
    """Sentiment analysis signal"""
    source: str
    sentiment_score: float  # -1 to 1
    confidence: float
    volume: int  # Number of mentions/articles
    trend: str  # "improving", "deteriorating", "stable"
    key_themes: List[str]
    risk_flags: List[str]


@dataclass
class MarketEvent:
    """Market-moving event detection"""
    event_type: str
    description: str
    impact_level: RiskLevel
    confidence: float
    detected_at: datetime
    related_tickers: List[str]
    sentiment_impact: float


class NewsAggregator:
    """Aggregate news from multiple sources"""
    
    def __init__(self):
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    async def get_news_for_symbol(self, symbol: str, hours_back: int = 24) -> List[NewsArticle]:
        """Get news articles for a specific symbol"""
        articles = []
        
        # RSS feeds (simplified list)
        rss_feeds = [
            f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US",
            "https://feeds.reuters.com/reuters/businessNews",
            "https://feeds.feedburner.com/CnbcBusiness"
        ]
        
        for feed_url in rss_feeds:
            try:
                feed_articles = await self._parse_rss_feed(feed_url, symbol, hours_back)
                articles.extend(feed_articles)
            except Exception as e:
                logger.error(f"Error parsing RSS feed {feed_url}: {e}")
                
        return articles
        
    async def _parse_rss_feed(self, feed_url: str, symbol: str, hours_back: int) -> List[NewsArticle]:
        """Parse RSS feed and filter relevant articles"""
        articles = []
        
        try:
            # Use feedparser in a thread to avoid blocking
            feed = await asyncio.to_thread(feedparser.parse, feed_url)
            
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            for entry in feed.entries:
                # Check if article is recent enough
                published_date = self._parse_date(entry.get('published', ''))
                if published_date and published_date < cutoff_time:
                    continue
                    
                # Check relevance to symbol
                title = entry.get('title', '')
                if self._is_relevant_to_symbol(title, symbol):
                    articles.append(NewsArticle(
                        title=title,
                        url=entry.get('link', ''),
                        published_date=published_date or datetime.now(),
                        source=feed.feed.get('title', 'Unknown'),
                        content=entry.get('summary', '')
                    ))
                    
        except Exception as e:
            logger.error(f"Error parsing RSS feed: {e}")
            
        return articles
        
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime"""
        try:
            # This is simplified - would need robust date parsing
            from dateutil import parser
            return parser.parse(date_str)
        except:
            return None
            
    def _is_relevant_to_symbol(self, text: str, symbol: str) -> bool:
        """Check if text is relevant to the symbol"""
        text_lower = text.lower()
        symbol_lower = symbol.lower()
        
        # Simple keyword matching - would be more sophisticated in production
        keywords = [symbol_lower]
        
        # Add company name if available (would come from a mapping)
        company_names = {
            'AAPL': ['apple', 'iphone', 'mac'],
            'GOOGL': ['google', 'alphabet', 'android'],
            'TSLA': ['tesla', 'elon musk', 'electric vehicle'],
            'MSFT': ['microsoft', 'azure', 'windows']
        }
        
        if symbol.upper() in company_names:
            keywords.extend(company_names[symbol.upper()])
            
        return any(keyword in text_lower for keyword in keywords)


class SentimentAnalyzer:
    """Analyze sentiment of news and social media"""
    
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.financial_sentiment_model = None
        self.tokenizer = None
        
    async def initialize(self):
        """Initialize ML models"""
        try:
            # Load financial sentiment model (FinBERT or similar)
            model_name = "ProsusAI/finbert"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.financial_sentiment_model = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=self.tokenizer,
                return_all_scores=True
            )
        except Exception as e:
            logger.warning(f"Could not load financial sentiment model: {e}")
            
    async def analyze_text_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text using multiple methods"""
        results = {}
        
        # VADER sentiment
        vader_scores = self.vader_analyzer.polarity_scores(text)
        results['vader'] = {
            'compound': vader_scores['compound'],
            'positive': vader_scores['pos'],
            'negative': vader_scores['neg'],
            'neutral': vader_scores['neu']
        }
        
        # TextBlob sentiment
        blob = TextBlob(text)
        results['textblob'] = {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
        
        # Financial sentiment model if available
        if self.financial_sentiment_model:
            try:
                # Truncate text if too long
                truncated_text = text[:512]
                fin_sentiment = await asyncio.to_thread(
                    self.financial_sentiment_model, truncated_text
                )
                
                # Convert to standardized format
                sentiment_map = {'positive': 1, 'negative': -1, 'neutral': 0}
                financial_score = 0
                
                for result in fin_sentiment[0]:
                    label = result['label'].lower()
                    score = result['score']
                    if label in sentiment_map:
                        financial_score += sentiment_map[label] * score
                        
                results['financial'] = {
                    'score': financial_score,
                    'raw_results': fin_sentiment[0]
                }
            except Exception as e:
                logger.error(f"Error in financial sentiment analysis: {e}")
                
        # Calculate composite sentiment
        composite_score = self._calculate_composite_sentiment(results)
        results['composite'] = composite_score
        
        return results
        
    def _calculate_composite_sentiment(self, results: Dict[str, Any]) -> float:
        """Calculate composite sentiment score from multiple methods"""
        scores = []
        weights = []
        
        # VADER (weight: 0.3)
        if 'vader' in results:
            scores.append(results['vader']['compound'])
            weights.append(0.3)
            
        # TextBlob (weight: 0.2)
        if 'textblob' in results:
            scores.append(results['textblob']['polarity'])
            weights.append(0.2)
            
        # Financial model (weight: 0.5 if available)
        if 'financial' in results:
            scores.append(results['financial']['score'])
            weights.append(0.5)
            
        if not scores:
            return 0.0
            
        # Weighted average
        weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
        total_weight = sum(weights)
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0


class EventDetector:
    """Detect market-moving events from news"""
    
    def __init__(self):
        self.event_keywords = {
            'earnings': ['earnings', 'quarterly results', 'financial results', 'Q1', 'Q2', 'Q3', 'Q4'],
            'merger': ['merger', 'acquisition', 'takeover', 'buyout'],
            'regulatory': ['SEC', 'FDA approval', 'regulation', 'investigation', 'lawsuit'],
            'management': ['CEO', 'CFO', 'resignation', 'appointed', 'leadership'],
            'product': ['product launch', 'new product', 'recall', 'approval'],
            'guidance': ['guidance', 'forecast', 'outlook', 'expects', 'projects']
        }
        
    async def detect_events(self, articles: List[NewsArticle]) -> List[MarketEvent]:
        """Detect market-moving events from news articles"""
        events = []
        
        for article in articles:
            try:
                detected_events = await self._analyze_article_for_events(article)
                events.extend(detected_events)
            except Exception as e:
                logger.error(f"Error detecting events in article: {e}")
                
        # Deduplicate and consolidate similar events
        consolidated_events = self._consolidate_events(events)
        
        return consolidated_events
        
    async def _analyze_article_for_events(self, article: NewsArticle) -> List[MarketEvent]:
        """Analyze single article for events"""
        events = []
        text = f"{article.title} {article.content or ''}"
        text_lower = text.lower()
        
        for event_type, keywords in self.event_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                # Determine impact level based on keywords and sentiment
                impact_level = self._determine_impact_level(text_lower, event_type)
                
                # Extract confidence based on keyword strength
                confidence = self._calculate_event_confidence(text_lower, keywords)
                
                events.append(MarketEvent(
                    event_type=event_type,
                    description=article.title,
                    impact_level=impact_level,
                    confidence=confidence,
                    detected_at=article.published_date,
                    related_tickers=[],  # Would extract from text
                    sentiment_impact=article.sentiment_score or 0.0
                ))
                
        return events
        
    def _determine_impact_level(self, text: str, event_type: str) -> RiskLevel:
        """Determine impact level of event"""
        high_impact_words = ['massive', 'major', 'significant', 'unprecedented', 'crisis']
        moderate_impact_words = ['notable', 'important', 'substantial']
        
        if any(word in text for word in high_impact_words):
            return RiskLevel.HIGH
        elif any(word in text for word in moderate_impact_words):
            return RiskLevel.MODERATE
        elif event_type in ['merger', 'regulatory']:
            return RiskLevel.HIGH
        else:
            return RiskLevel.LOW
            
    def _calculate_event_confidence(self, text: str, keywords: List[str]) -> float:
        """Calculate confidence in event detection"""
        keyword_count = sum(1 for keyword in keywords if keyword in text)
        max_confidence = min(keyword_count * 0.3, 1.0)
        return max(max_confidence, 0.3)  # Minimum 30% confidence
        
    def _consolidate_events(self, events: List[MarketEvent]) -> List[MarketEvent]:
        """Consolidate similar events"""
        # Group events by type and time window
        consolidated = []
        event_groups = defaultdict(list)
        
        for event in events:
            key = (event.event_type, event.detected_at.date())
            event_groups[key].append(event)
            
        for group in event_groups.values():
            if len(group) == 1:
                consolidated.append(group[0])
            else:
                # Merge similar events
                merged_event = self._merge_events(group)
                consolidated.append(merged_event)
                
        return consolidated
        
    def _merge_events(self, events: List[MarketEvent]) -> MarketEvent:
        """Merge similar events"""
        # Use the event with highest confidence as base
        base_event = max(events, key=lambda e: e.confidence)
        
        # Average confidence
        avg_confidence = sum(e.confidence for e in events) / len(events)
        
        # Combine descriptions
        descriptions = [e.description for e in events]
        combined_description = f"Multiple {base_event.event_type} events: {'; '.join(descriptions[:3])}"
        
        return MarketEvent(
            event_type=base_event.event_type,
            description=combined_description,
            impact_level=base_event.impact_level,
            confidence=avg_confidence,
            detected_at=base_event.detected_at,
            related_tickers=base_event.related_tickers,
            sentiment_impact=sum(e.sentiment_impact for e in events) / len(events)
        )


class SentimentAnalysisAgent(BaseAgent):
    """Agent for sentiment analysis and event detection"""
    
    def __init__(self, message_bus, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.SENTIMENT_ANALYSIS, message_bus, config)
        
        self.news_aggregator = NewsAggregator()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.event_detector = EventDetector()
        
        # Configuration
        self.news_hours_back = config.get('news_hours_back', 24)
        self.sentiment_threshold = config.get('sentiment_threshold', 0.3)
        
    async def _initialize(self):
        """Initialize the sentiment analysis agent"""
        await self.sentiment_analyzer.initialize()
        logger.info("Sentiment Analysis Agent initialized")
        
    async def _cleanup(self):
        """Cleanup resources"""
        pass
        
    async def process_message(self, message: AgentMessage) -> Optional[AgentResponse]:
        """Process sentiment analysis requests"""
        
        if message.message_type == MessageType.ANALYSIS_REQUEST:
            return await self._handle_analysis_request(message)
            
        return None
        
    async def _handle_analysis_request(self, message: AgentMessage) -> AgentResponse:
        """Handle sentiment analysis requests"""
        try:
            analysis_type = message.payload.get('analysis_type')
            symbol = message.payload.get('symbol')
            
            if not symbol:
                return AgentResponse(
                    success=False,
                    error="Symbol is required for sentiment analysis",
                    confidence=0.0
                )
                
            # Perform sentiment analysis
            analysis_result = await self._perform_sentiment_analysis(symbol)
            
            return AgentResponse(
                success=True,
                data=analysis_result,
                confidence=analysis_result.get('overall_confidence', 0.5),
                metadata={'symbol': symbol, 'analysis_type': analysis_type}
            )
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return AgentResponse(
                success=False,
                error=str(e),
                confidence=0.0
            )
            
    async def _perform_sentiment_analysis(self, symbol: str) -> Dict[str, Any]:
        """Perform comprehensive sentiment analysis"""
        
        # Get news articles
        async with self.news_aggregator:
            articles = await self.news_aggregator.get_news_for_symbol(symbol, self.news_hours_back)
            
        # Analyze sentiment of each article
        for article in articles:
            if article.content:
                sentiment_result = await self.sentiment_analyzer.analyze_text_sentiment(
                    f"{article.title} {article.content}"
                )
                article.sentiment_score = sentiment_result['composite']
                
        # Detect events
        events = await self.event_detector.detect_events(articles)
        
        # Calculate aggregate sentiment signals
        sentiment_signals = await self._calculate_sentiment_signals(articles)
        
        # Generate recommendation
        recommendation = await self._generate_sentiment_recommendation(
            sentiment_signals, events, articles
        )
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'news_articles': [asdict(article) for article in articles],
            'detected_events': [asdict(event) for event in events],
            'sentiment_signals': asdict(sentiment_signals),
            'recommendation': recommendation,
            'overall_confidence': recommendation.get('confidence', 0.5)
        }
        
    async def _calculate_sentiment_signals(self, articles: List[NewsArticle]) -> SentimentSignal:
        """Calculate aggregate sentiment signals"""
        
        if not articles:
            return SentimentSignal(
                source="news",
                sentiment_score=0.0,
                confidence=0.0,
                volume=0,
                trend="stable",
                key_themes=[],
                risk_flags=[]
            )
            
        # Calculate weighted sentiment (recent articles have higher weight)
        now = datetime.now()
        weighted_sentiment = 0
        total_weight = 0
        
        for article in articles:
            if article.sentiment_score is not None:
                # Weight by recency (max 24 hours)
                hours_ago = (now - article.published_date).total_seconds() / 3600
                weight = max(0.1, 1.0 - (hours_ago / 24))
                
                weighted_sentiment += article.sentiment_score * weight
                total_weight += weight
                
        avg_sentiment = weighted_sentiment / total_weight if total_weight > 0 else 0.0
        
        # Determine trend (compare last 12 hours vs previous 12 hours)
        cutoff = now - timedelta(hours=12)
        recent_articles = [a for a in articles if a.published_date > cutoff]
        older_articles = [a for a in articles if a.published_date <= cutoff]
        
        recent_sentiment = np.mean([a.sentiment_score for a in recent_articles if a.sentiment_score is not None]) if recent_articles else 0
        older_sentiment = np.mean([a.sentiment_score for a in older_articles if a.sentiment_score is not None]) if older_articles else 0
        
        trend = "stable"
        if recent_sentiment > older_sentiment + 0.1:
            trend = "improving"
        elif recent_sentiment < older_sentiment - 0.1:
            trend = "deteriorating"
            
        # Extract key themes (simplified)
        key_themes = self._extract_key_themes(articles)
        
        # Identify risk flags
        risk_flags = self._identify_risk_flags(articles)
        
        # Calculate confidence based on volume and consistency
        sentiment_std = np.std([a.sentiment_score for a in articles if a.sentiment_score is not None])
        volume_factor = min(len(articles) / 10, 1.0)  # More articles = higher confidence
        consistency_factor = max(0.3, 1.0 - sentiment_std) if sentiment_std else 0.5
        confidence = volume_factor * consistency_factor
        
        return SentimentSignal(
            source="news",
            sentiment_score=avg_sentiment,
            confidence=confidence,
            volume=len(articles),
            trend=trend,
            key_themes=key_themes,
            risk_flags=risk_flags
        )
        
    def _extract_key_themes(self, articles: List[NewsArticle]) -> List[str]:
        """Extract key themes from articles"""
        # Simplified theme extraction
        themes = defaultdict(int)
        common_terms = [
            'earnings', 'revenue', 'growth', 'profit', 'loss', 'merger', 'acquisition',
            'product', 'market', 'competition', 'regulation', 'approval', 'partnership'
        ]
        
        for article in articles:
            text = f"{article.title} {article.content or ''}".lower()
            for term in common_terms:
                if term in text:
                    themes[term] += 1
                    
        # Return top themes
        sorted_themes = sorted(themes.items(), key=lambda x: x[1], reverse=True)
        return [theme for theme, count in sorted_themes[:5]]
        
    def _identify_risk_flags(self, articles: List[NewsArticle]) -> List[str]:
        """Identify risk flags from articles"""
        risk_flags = []
        risk_keywords = {
            'legal': ['lawsuit', 'investigation', 'SEC', 'fraud', 'violation'],
            'financial': ['bankruptcy', 'debt', 'loss', 'decline', 'miss'],
            'operational': ['recall', 'shutdown', 'layoff', 'restructuring'],
            'market': ['competition', 'market share', 'disruption']
        }
        
        for article in articles:
            text = f"{article.title} {article.content or ''}".lower()
            for risk_type, keywords in risk_keywords.items():
                if any(keyword in text for keyword in keywords):
                    risk_flags.append(f"{risk_type}_risk")
                    
        return list(set(risk_flags))  # Remove duplicates
        
    async def _generate_sentiment_recommendation(self, 
                                               sentiment_signals: SentimentSignal,
                                               events: List[MarketEvent],
                                               articles: List[NewsArticle]) -> Dict[str, Any]:
        """Generate recommendation based on sentiment analysis"""
        
        # Base sentiment signal
        sentiment_score = sentiment_signals.sentiment_score
        confidence = sentiment_signals.confidence
        
        # Adjust for events
        event_impact = 0
        high_impact_events = [e for e in events if e.impact_level in [RiskLevel.HIGH, RiskLevel.EXTREME]]
        
        if high_impact_events:
            # Average event sentiment impact
            event_impact = np.mean([e.sentiment_impact for e in high_impact_events])
            confidence *= 1.2  # Higher confidence when significant events detected
            
        # Combined sentiment
        combined_sentiment = (sentiment_score + event_impact) / 2 if event_impact != 0 else sentiment_score
        
        # Adjust for trend
        trend_adjustment = 0
        if sentiment_signals.trend == "improving":
            trend_adjustment = 0.1
        elif sentiment_signals.trend == "deteriorating":
            trend_adjustment = -0.1
            
        final_sentiment = combined_sentiment + trend_adjustment
        
        # Generate action recommendation
        if final_sentiment > 0.3 and confidence > 0.6:
            action = ActionType.BUY.value
            risk_level = RiskLevel.LOW
        elif final_sentiment < -0.3 and confidence > 0.6:
            action = ActionType.SELL.value
            risk_level = RiskLevel.MODERATE
        else:
            action = ActionType.HOLD.value
            risk_level = RiskLevel.LOW
            
        # Adjust for risk flags
        if sentiment_signals.risk_flags:
            if action == ActionType.BUY.value:
                action = ActionType.HOLD.value
            risk_level = RiskLevel.HIGH
            confidence *= 0.8
            
        return {
            'action': action,
            'confidence': min(confidence, 1.0),
            'sentiment_score': final_sentiment,
            'risk_level': risk_level.value,
            'trend': sentiment_signals.trend,
            'news_volume': len(articles),
            'events_detected': len(events),
            'risk_flags': sentiment_signals.risk_flags,
            'reasoning': f"Sentiment analysis shows {final_sentiment:.2f} sentiment with {sentiment_signals.trend} trend"
        }