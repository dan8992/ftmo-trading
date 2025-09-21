#!/usr/bin/env python3
"""
Technical Pattern Recognition Service
Analyzes 1-minute OHLC data for technical patterns and generates trading signals
"""
import asyncio
import psycopg2
import pandas as pd
import numpy as np
import talib
from datetime import datetime, timedelta
import logging
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TechnicalSignal:
    timestamp: datetime
    currency_pair: str
    signal_type: str  # BUY, SELL, HOLD
    confidence: float
    reasoning: str
    indicators: Dict
    risk_score: float

class TechnicalPatternService:
    def __init__(self):
        self.db_config = {
            'host': os.getenv('POSTGRES_HOST', 'postgres'),
            'port': int(os.getenv('POSTGRES_PORT', 5432)),
            'database': os.getenv('POSTGRES_DB', 'finrl_dax'),
            'user': os.getenv('POSTGRES_USER', 'finrl_user'),
            'password': os.getenv('POSTGRES_PASSWORD')
        }
        
        # Technical analysis parameters
        self.lookback_periods = {
            'short': 50,   # 50 minutes
            'medium': 200, # ~3 hours
            'long': 480    # 8 hours
        }

    async def get_ohlc_data(self, symbol: str, lookback_minutes: int = 480) -> pd.DataFrame:
        """Fetch OHLC data for technical analysis"""
        conn = psycopg2.connect(**self.db_config)
        try:
            query = """
                SELECT timestamp, open, high, low, close, volume
                FROM trading_data_1m 
                WHERE symbol = %s 
                    AND timestamp > %s
                ORDER BY timestamp ASC
            """
            
            cutoff_time = datetime.now() - timedelta(minutes=lookback_minutes)
            df = pd.read_sql_query(query, conn, params=(symbol, cutoff_time))
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                
            return df
            
        finally:
            conn.close()

    def calculate_technical_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate comprehensive technical indicators"""
        if len(df) < 50:
            return {}
        
        indicators = {}
        
        # Price data
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        volume = df['volume'].values
        
        try:
            # Moving Averages
            indicators['sma_20'] = talib.SMA(close, timeperiod=20)
            indicators['sma_50'] = talib.SMA(close, timeperiod=50)
            indicators['ema_12'] = talib.EMA(close, timeperiod=12)
            indicators['ema_26'] = talib.EMA(close, timeperiod=26)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            indicators['macd'] = macd
            indicators['macd_signal'] = macd_signal
            indicators['macd_histogram'] = macd_hist
            
            # RSI
            indicators['rsi'] = talib.RSI(close, timeperiod=14)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            indicators['bb_upper'] = bb_upper
            indicators['bb_middle'] = bb_middle
            indicators['bb_lower'] = bb_lower
            
            # Stochastic
            slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
            indicators['stoch_k'] = slowk
            indicators['stoch_d'] = slowd
            
            # ADX (Trend Strength)
            indicators['adx'] = talib.ADX(high, low, close, timeperiod=14)
            
            # ATR (Volatility)
            indicators['atr'] = talib.ATR(high, low, close, timeperiod=14)
            
            # Volume indicators
            indicators['obv'] = talib.OBV(close, volume)
            
            # Support/Resistance levels
            indicators['resistance'] = self.calculate_resistance_levels(df)
            indicators['support'] = self.calculate_support_levels(df)
            
        except Exception as e:
            logger.warning(f"Error calculating technical indicators: {e}")
        
        return indicators

    def calculate_resistance_levels(self, df: pd.DataFrame, window: int = 20) -> List[float]:
        """Calculate resistance levels from recent highs"""
        highs = df['high'].rolling(window=window).max()
        resistance_levels = highs.dropna().tail(5).tolist()
        return sorted(set(resistance_levels), reverse=True)[:3]

    def calculate_support_levels(self, df: pd.DataFrame, window: int = 20) -> List[float]:
        """Calculate support levels from recent lows"""
        lows = df['low'].rolling(window=window).min()
        support_levels = lows.dropna().tail(5).tolist()
        return sorted(set(support_levels))[:3]

    def analyze_trend_patterns(self, df: pd.DataFrame, indicators: Dict) -> Dict:
        """Analyze trend patterns and momentum"""
        patterns = {
            'trend_direction': 'NEUTRAL',
            'trend_strength': 0.0,
            'momentum': 'NEUTRAL',
            'volatility': 'NORMAL'
        }
        
        if len(df) < 50 or not indicators:
            return patterns
        
        try:
            current_price = df['close'].iloc[-1]
            
            # Trend Direction Analysis
            sma_20 = indicators.get('sma_20', [])
            sma_50 = indicators.get('sma_50', [])
            
            if len(sma_20) > 0 and len(sma_50) > 0:
                if not np.isnan(sma_20[-1]) and not np.isnan(sma_50[-1]):
                    if current_price > sma_20[-1] > sma_50[-1]:
                        patterns['trend_direction'] = 'BULLISH'
                    elif current_price < sma_20[-1] < sma_50[-1]:
                        patterns['trend_direction'] = 'BEARISH'
            
            # ADX for trend strength
            adx = indicators.get('adx', [])
            if len(adx) > 0 and not np.isnan(adx[-1]):
                if adx[-1] > 25:
                    patterns['trend_strength'] = min(adx[-1] / 50.0, 1.0)
                else:
                    patterns['trend_strength'] = 0.0
            
            # MACD for momentum
            macd = indicators.get('macd', [])
            macd_signal = indicators.get('macd_signal', [])
            
            if len(macd) > 0 and len(macd_signal) > 0:
                if not np.isnan(macd[-1]) and not np.isnan(macd_signal[-1]):
                    if macd[-1] > macd_signal[-1]:
                        patterns['momentum'] = 'BULLISH'
                    elif macd[-1] < macd_signal[-1]:
                        patterns['momentum'] = 'BEARISH'
            
            # ATR for volatility
            atr = indicators.get('atr', [])
            if len(atr) > 0 and not np.isnan(atr[-1]):
                atr_pct = atr[-1] / current_price * 100
                if atr_pct > 0.5:
                    patterns['volatility'] = 'HIGH'
                elif atr_pct < 0.2:
                    patterns['volatility'] = 'LOW'
                    
        except Exception as e:
            logger.warning(f"Error analyzing trend patterns: {e}")
        
        return patterns

    def generate_trading_signal(self, df: pd.DataFrame, indicators: Dict, patterns: Dict) -> TechnicalSignal:
        """Generate trading signal based on technical analysis"""
        current_time = datetime.now()
        currency_pair = 'EURUSD'  # Will be passed as parameter
        
        if len(df) < 50 or not indicators:
            return TechnicalSignal(
                timestamp=current_time,
                currency_pair=currency_pair,
                signal_type='HOLD',
                confidence=0.0,
                reasoning='Insufficient data for analysis',
                indicators={},
                risk_score=1.0
            )
        
        signal_strength = 0.0
        reasoning_parts = []
        risk_factors = []
        
        try:
            current_price = df['close'].iloc[-1]
            
            # RSI Analysis
            rsi = indicators.get('rsi', [])
            if len(rsi) > 0 and not np.isnan(rsi[-1]):
                if rsi[-1] < 30:
                    signal_strength += 0.3
                    reasoning_parts.append(f"RSI oversold ({rsi[-1]:.1f})")
                elif rsi[-1] > 70:
                    signal_strength -= 0.3
                    reasoning_parts.append(f"RSI overbought ({rsi[-1]:.1f})")
            
            # MACD Analysis
            macd = indicators.get('macd', [])
            macd_signal = indicators.get('macd_signal', [])
            if len(macd) > 1 and len(macd_signal) > 1:
                if not np.isnan(macd[-1]) and not np.isnan(macd_signal[-1]):
                    # MACD crossover
                    if macd[-2] <= macd_signal[-2] and macd[-1] > macd_signal[-1]:
                        signal_strength += 0.4
                        reasoning_parts.append("MACD bullish crossover")
                    elif macd[-2] >= macd_signal[-2] and macd[-1] < macd_signal[-1]:
                        signal_strength -= 0.4
                        reasoning_parts.append("MACD bearish crossover")
            
            # Bollinger Bands Analysis
            bb_upper = indicators.get('bb_upper', [])
            bb_lower = indicators.get('bb_lower', [])
            if len(bb_upper) > 0 and len(bb_lower) > 0:
                if not np.isnan(bb_upper[-1]) and not np.isnan(bb_lower[-1]):
                    if current_price <= bb_lower[-1]:
                        signal_strength += 0.2
                        reasoning_parts.append("Price at lower Bollinger Band")
                    elif current_price >= bb_upper[-1]:
                        signal_strength -= 0.2
                        reasoning_parts.append("Price at upper Bollinger Band")
            
            # Trend confirmation
            if patterns['trend_direction'] == 'BULLISH' and signal_strength > 0:
                signal_strength += 0.2 * patterns['trend_strength']
                reasoning_parts.append(f"Bullish trend confirmation (strength: {patterns['trend_strength']:.2f})")
            elif patterns['trend_direction'] == 'BEARISH' and signal_strength < 0:
                signal_strength -= 0.2 * patterns['trend_strength']
                reasoning_parts.append(f"Bearish trend confirmation (strength: {patterns['trend_strength']:.2f})")
            
            # Risk assessment
            risk_score = 0.5  # Base risk
            
            if patterns['volatility'] == 'HIGH':
                risk_score += 0.3
                risk_factors.append("High volatility")
            elif patterns['volatility'] == 'LOW':
                risk_score -= 0.1
                
            if patterns['trend_strength'] < 0.3:
                risk_score += 0.2
                risk_factors.append("Weak trend")
            
            # Determine signal type
            if signal_strength > 0.3:
                signal_type = 'BUY'
            elif signal_strength < -0.3:
                signal_type = 'SELL'
            else:
                signal_type = 'HOLD'
            
            confidence = min(abs(signal_strength), 1.0)
            
            # Compile reasoning
            reasoning = '; '.join(reasoning_parts) if reasoning_parts else 'No clear signal'
            if risk_factors:
                reasoning += f' | Risk factors: {", ".join(risk_factors)}'
            
            # Prepare indicators summary
            indicators_summary = {
                'rsi': rsi[-1] if len(rsi) > 0 and not np.isnan(rsi[-1]) else None,
                'macd': macd[-1] if len(macd) > 0 and not np.isnan(macd[-1]) else None,
                'trend_direction': patterns['trend_direction'],
                'trend_strength': patterns['trend_strength'],
                'volatility': patterns['volatility'],
                'signal_strength': signal_strength
            }
            
            return TechnicalSignal(
                timestamp=current_time,
                currency_pair=currency_pair,
                signal_type=signal_type,
                confidence=confidence,
                reasoning=reasoning,
                indicators=indicators_summary,
                risk_score=min(max(risk_score, 0.0), 1.0)
            )
            
        except Exception as e:
            logger.error(f"Error generating trading signal: {e}")
            return TechnicalSignal(
                timestamp=current_time,
                currency_pair=currency_pair,
                signal_type='HOLD',
                confidence=0.0,
                reasoning=f'Error in signal generation: {str(e)}',
                indicators={},
                risk_score=1.0
            )

    async def analyze_currency_pair(self, symbol: str) -> Optional[TechnicalSignal]:
        """Perform complete technical analysis for a currency pair"""
        try:
            # Fetch OHLC data
            df = await self.get_ohlc_data(symbol, lookback_minutes=480)
            
            if df.empty:
                logger.warning(f"No data available for {symbol}")
                return None
            
            # Calculate technical indicators
            indicators = self.calculate_technical_indicators(df)
            
            # Analyze patterns
            patterns = self.analyze_trend_patterns(df, indicators)
            
            # Generate signal
            signal = self.generate_trading_signal(df, indicators, patterns)
            signal.currency_pair = symbol
            
            logger.info(f"Generated signal for {symbol}: {signal.signal_type} (confidence: {signal.confidence:.2f})")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None

    async def run_technical_analysis_cycle(self):
        """Run technical analysis for all monitored currency pairs"""
        currency_pairs = ['EURUSD', 'GBPUSD']
        signals = []
        
        for pair in currency_pairs:
            signal = await self.analyze_currency_pair(pair)
            if signal:
                signals.append(signal)
        
        return signals

    async def run_continuous_technical_analysis(self, interval_minutes: int = 1):
        """Run continuous technical analysis"""
        while True:
            try:
                signals = await self.run_technical_analysis_cycle()
                logger.info(f"Generated {len(signals)} technical signals")
                
                # Signals will be consumed by the main signal generation engine
                await asyncio.sleep(interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"Error in technical analysis cycle: {e}")
                await asyncio.sleep(60)

if __name__ == "__main__":
    service = TechnicalPatternService()
    asyncio.run(service.run_continuous_technical_analysis())