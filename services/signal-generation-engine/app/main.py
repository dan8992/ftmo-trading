#!/usr/bin/env python3
"""
LLM Signal Generation Engine
Combines sentiment analysis, technical patterns, and economic events to generate trading signals
"""
import asyncio
import psycopg2
import json
from datetime import datetime, timedelta
import logging
import os
from typing import Dict, List, Optional
import numpy as np

# Import our services
from finbert_sentiment_service import FinBERTSentimentService
from technical_pattern_service import TechnicalPatternService, TechnicalSignal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignalGenerationEngine:
    def __init__(self):
        self.db_config = {
            'host': os.getenv('POSTGRES_HOST', 'postgres'),
            'port': int(os.getenv('POSTGRES_PORT', 5432)),
            'database': os.getenv('POSTGRES_DB', 'finrl_dax'),
            'user': os.getenv('POSTGRES_USER', 'finrl_user'),
            'password': os.getenv('POSTGRES_PASSWORD')
        }
        
        # Initialize component services
        self.sentiment_service = FinBERTSentimentService()
        self.technical_service = TechnicalPatternService()
        
        # Signal fusion weights
        self.fusion_weights = {
            'technical': 0.6,
            'sentiment': 0.3,
            'economic': 0.1
        }
        
        # Risk management parameters
        self.max_signal_confidence = 0.95  # Never be 100% confident
        self.min_signal_threshold = 0.15   # Minimum confidence for actionable signals

    async def initialize_services(self):
        """Initialize all component services"""
        logger.info("Initializing signal generation services...")
        await self.sentiment_service.initialize_model()
        logger.info("Signal generation engine ready")

    async def get_economic_events_impact(self, currency_pair: str, hours_ahead: int = 24) -> Dict:
        """Analyze upcoming economic events impact"""
        conn = psycopg2.connect(**self.db_config)
        try:
            # Get upcoming high/medium impact events
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT title, country, currency, impact, event_time, forecast, previous
                    FROM economic_events 
                    WHERE currency IN %s
                        AND impact IN ('High', 'Medium')
                        AND event_time BETWEEN %s AND %s
                        AND processed = FALSE
                    ORDER BY event_time ASC
                """, (
                    tuple(currency_pair.replace('USD', '').split() + ['USD']),
                    datetime.now(),
                    datetime.now() + timedelta(hours=hours_ahead)
                ))
                
                events = cur.fetchall()
                
                if not events:
                    return {'impact_score': 0.0, 'risk_level': 'LOW', 'events_count': 0}
                
                # Calculate impact score
                impact_score = 0.0
                high_impact_count = 0
                
                for event in events:
                    title, country, currency, impact, event_time, forecast, previous = event
                    
                    if impact == 'High':
                        impact_score += 0.3
                        high_impact_count += 1
                    elif impact == 'Medium':
                        impact_score += 0.1
                
                # Determine risk level
                if high_impact_count >= 2 or impact_score >= 0.5:
                    risk_level = 'HIGH'
                elif high_impact_count >= 1 or impact_score >= 0.2:
                    risk_level = 'MEDIUM'
                else:
                    risk_level = 'LOW'
                
                return {
                    'impact_score': min(impact_score, 1.0),
                    'risk_level': risk_level,
                    'events_count': len(events),
                    'high_impact_count': high_impact_count
                }
                
        finally:
            conn.close()

    def fuse_signals(self, technical_signal: TechnicalSignal, sentiment_data: Dict, economic_data: Dict) -> Dict:
        """Fuse multiple signal sources into final trading signal"""
        
        # Technical signal component
        technical_score = 0.0
        if technical_signal.signal_type == 'BUY':
            technical_score = technical_signal.confidence
        elif technical_signal.signal_type == 'SELL':
            technical_score = -technical_signal.confidence
        
        # Sentiment component
        sentiment_score = sentiment_data.get('avg_sentiment', 0.0)
        sentiment_confidence = min(sentiment_data.get('article_count', 0) / 10.0, 1.0)
        
        # Economic events component
        economic_score = 0.0
        if economic_data['risk_level'] == 'HIGH':
            economic_score = -0.3  # High risk events create uncertainty
        elif economic_data['risk_level'] == 'MEDIUM':
            economic_score = -0.1
        
        # Weighted fusion
        fused_score = (
            self.fusion_weights['technical'] * technical_score +
            self.fusion_weights['sentiment'] * sentiment_score * sentiment_confidence +
            self.fusion_weights['economic'] * economic_score
        )
        
        # Determine final signal
        if fused_score > self.min_signal_threshold:
            signal_type = 'BUY'
            confidence = min(fused_score, self.max_signal_confidence)
        elif fused_score < -self.min_signal_threshold:
            signal_type = 'SELL'
            confidence = min(abs(fused_score), self.max_signal_confidence)
        else:
            signal_type = 'HOLD'
            confidence = abs(fused_score)
        
        # Calculate risk score
        base_risk = technical_signal.risk_score
        
        # Adjust risk based on sentiment volatility
        if sentiment_data.get('sentiment_volatility', 0) > 0.5:
            base_risk += 0.2
            
        # Adjust risk based on economic events
        if economic_data['risk_level'] == 'HIGH':
            base_risk += 0.3
        elif economic_data['risk_level'] == 'MEDIUM':
            base_risk += 0.1
        
        risk_score = min(max(base_risk, 0.0), 1.0)
        
        # Compile reasoning
        reasoning_parts = [technical_signal.reasoning]
        
        if sentiment_data.get('article_count', 0) > 0:
            reasoning_parts.append(f"Sentiment: {sentiment_score:.2f} from {sentiment_data['article_count']} articles")
        
        if economic_data['events_count'] > 0:
            reasoning_parts.append(f"Economic events: {economic_data['events_count']} upcoming ({economic_data['risk_level']} risk)")
        
        # Prepare technical indicators for storage
        technical_indicators = technical_signal.indicators.copy()
        technical_indicators.update({
            'sentiment_score': sentiment_score,
            'sentiment_confidence': sentiment_confidence,
            'economic_risk': economic_data['risk_level'],
            'fusion_score': fused_score,
            'fusion_weights': self.fusion_weights
        })
        
        return {
            'timestamp': datetime.now(),
            'currency_pair': technical_signal.currency_pair,
            'signal_type': signal_type,
            'confidence': confidence,
            'reasoning': ' | '.join(reasoning_parts),
            'technical_indicators': technical_indicators,
            'risk_score': risk_score
        }

    async def store_trading_signal(self, signal_data: Dict):
        """Store generated trading signal in database"""
        conn = psycopg2.connect(**self.db_config)
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO trading_signals 
                    (timestamp, symbol, signal_type, confidence, reasoning, technical_indicators, risk_score)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    signal_data['timestamp'],
                    signal_data['currency_pair'],
                    signal_data['signal_type'],
                    signal_data['confidence'],
                    signal_data['reasoning'],
                    json.dumps(signal_data['technical_indicators']),
                    signal_data['risk_score']
                ))
                conn.commit()
                logger.info(f"Stored {signal_data['signal_type']} signal for {signal_data['currency_pair']} (confidence: {signal_data['confidence']:.2f})")
        finally:
            conn.close()

    async def generate_signal_for_pair(self, currency_pair: str) -> Optional[Dict]:
        """Generate comprehensive trading signal for a currency pair"""
        try:
            logger.info(f"Generating signal for {currency_pair}")
            
            # Get technical analysis signal
            technical_signal = await self.technical_service.analyze_currency_pair(currency_pair)
            if not technical_signal:
                logger.warning(f"No technical signal generated for {currency_pair}")
                return None
            
            # Get sentiment analysis
            sentiment_data = await self.sentiment_service.calculate_aggregate_sentiment(currency_pair, hours_back=24)
            
            # Get economic events impact
            economic_data = await self.get_economic_events_impact(currency_pair, hours_ahead=24)
            
            # Fuse all signals
            fused_signal = self.fuse_signals(technical_signal, sentiment_data, economic_data)
            
            # Store the signal
            await self.store_trading_signal(fused_signal)
            
            return fused_signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {currency_pair}: {e}")
            return None

    async def run_signal_generation_cycle(self):
        """Run complete signal generation cycle for all monitored pairs"""
        currency_pairs = ['EURUSD', 'GBPUSD']
        generated_signals = []
        
        # Process sentiment analysis first
        await self.sentiment_service.process_sentiment_batch()
        
        # Generate signals for each pair
        for pair in currency_pairs:
            signal = await self.generate_signal_for_pair(pair)
            if signal:
                generated_signals.append(signal)
        
        logger.info(f"Generated {len(generated_signals)} trading signals")
        return generated_signals

    async def get_latest_signals(self, currency_pair: str = None, limit: int = 10) -> List[Dict]:
        """Get latest trading signals from database"""
        conn = psycopg2.connect(**self.db_config)
        try:
            with conn.cursor() as cur:
                if currency_pair:
                    cur.execute("""
                        SELECT timestamp, symbol, signal_type, confidence, reasoning, technical_indicators, risk_score
                        FROM trading_signals 
                        WHERE symbol = %s
                        ORDER BY timestamp DESC
                        LIMIT %s
                    """, (currency_pair, limit))
                else:
                    cur.execute("""
                        SELECT timestamp, symbol, signal_type, confidence, reasoning, technical_indicators, risk_score
                        FROM trading_signals 
                        ORDER BY timestamp DESC
                        LIMIT %s
                    """, (limit,))
                
                columns = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
                
                signals = []
                for row in rows:
                    signal = dict(zip(columns, row))
                    if signal['technical_indicators']:
                        signal['technical_indicators'] = json.loads(signal['technical_indicators'])
                    signals.append(signal)
                
                return signals
                
        finally:
            conn.close()

    async def run_continuous_signal_generation(self, interval_minutes: int = 5):
        """Run continuous signal generation"""
        await self.initialize_services()
        
        logger.info(f"Starting continuous signal generation (interval: {interval_minutes} minutes)")
        
        while True:
            try:
                await self.run_signal_generation_cycle()
                await asyncio.sleep(interval_minutes * 60)
            except Exception as e:
                logger.error(f"Error in signal generation cycle: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry

    async def backtest_signals(self, start_date: datetime, end_date: datetime, currency_pair: str = 'EURUSD'):
        """Backtest signal generation performance"""
        logger.info(f"Starting backtest for {currency_pair} from {start_date} to {end_date}")
        
        # This would implement backtesting logic
        # For now, we'll simulate the process
        
        conn = psycopg2.connect(**self.db_config)
        try:
            with conn.cursor() as cur:
                # Get historical price data for the period
                cur.execute("""
                    SELECT COUNT(*) as price_points
                    FROM trading_data_1m 
                    WHERE symbol = %s 
                        AND timestamp BETWEEN %s AND %s
                """, (currency_pair, start_date, end_date))
                
                result = cur.fetchone()
                price_points = result[0] if result else 0
                
                logger.info(f"Backtest would analyze {price_points} price points")
                
                return {
                    'currency_pair': currency_pair,
                    'start_date': start_date,
                    'end_date': end_date,
                    'price_points': price_points,
                    'status': 'simulated'
                }
                
        finally:
            conn.close()

if __name__ == "__main__":
    engine = SignalGenerationEngine()
    
    # Run continuous signal generation
    asyncio.run(engine.run_continuous_signal_generation())