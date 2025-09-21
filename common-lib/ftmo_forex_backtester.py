import os
#!/usr/bin/env python3
"""
FTMO Challenge Forex Backtesting Framework
Uses 1-minute EURUSD/GBPUSD data for precise FTMO rule simulation
Professional implementation for 30-year trader standards
"""

import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime, timedelta, time
import warnings
import json
import pytz
warnings.filterwarnings('ignore')

class FTMOForexBacktester:
    def __init__(self, initial_balance=100000, currency='USD'):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.equity = initial_balance  # Real-time equity including floating P&L
        self.peak_equity = initial_balance
        self.daily_starting_balance = initial_balance
        self.trades = []
        self.positions = {}
        self.daily_metrics = {}
        self.trading_days = 0
        self.currency = currency
        
        # FTMO Challenge Rules
        self.profit_target = 0.10  # 10%
        self.max_daily_loss = 0.05  # 5% of initial balance
        self.max_total_loss = 0.10  # 10% of initial balance
        self.min_trading_days = 4
        
        # Professional Risk Management (30-year trader standards)
        self.max_risk_per_trade = 0.01  # 1% risk per trade
        self.max_concurrent_positions = 2  # Conservative for FTMO
        self.max_daily_trades = 5  # Limit overtrading
        self.max_correlation_exposure = 0.02  # 2% max on correlated pairs
        
        # Trading session filters (London/NY overlap for best liquidity)
        self.trading_start_hour = 8   # 8:00 GMT
        self.trading_end_hour = 17    # 17:00 GMT
        self.avoid_news_minutes = 30  # Minutes to avoid around major news
        
        # State tracking
        self.current_date = None
        self.daily_trades_count = 0
        self.challenge_failed = False
        self.failure_reason = ""
        
    def connect_to_db(self):
        """Connect to PostgreSQL database"""
        try:
            conn = psycopg2.connect(
                host="postgres-service.dax-trading.svc.cluster.local",
                database="finrl_dax",
                user="finrl_user", 
                password=os.getenv("POSTGRES_PASSWORD"),
                port="5432"
            )
            return conn
        except:
            print("Database connection failed")
            return None
    
    def get_forex_data(self, symbol, start_date, end_date, limit=None):
        """Fetch 1-minute forex data from database"""
        conn = self.connect_to_db()
        if not conn:
            return pd.DataFrame()
        
        try:
            # Build query with optional limit for testing
            limit_clause = f"LIMIT {limit}" if limit else ""
            
            query = f"""
            SELECT timestamp, open, high, low, close, volume
            FROM trading_data_1m 
            WHERE symbol = %s 
              AND timestamp BETWEEN %s AND %s
            ORDER BY timestamp
            {limit_clause}
            """
            
            df = pd.read_sql_query(query, conn, params=(symbol, start_date, end_date))
            conn.close()
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                # Add derived columns
                df['hour'] = df.index.hour
                df['minute'] = df.index.minute
                df['weekday'] = df.index.weekday
                
            return df
            
        except Exception as e:
            print(f"Error fetching {symbol} data: {e}")
            conn.close()
            return pd.DataFrame()
    
    def is_trading_session(self, timestamp):
        """Check if timestamp is within preferred trading hours"""
        hour = timestamp.hour
        weekday = timestamp.weekday()
        
        # Avoid weekends
        if weekday >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # London/NY overlap (8:00-17:00 GMT) - best liquidity
        return self.trading_start_hour <= hour < self.trading_end_hour
    
    def calculate_position_size(self, entry_price, stop_loss_price, symbol):
        """Calculate position size using professional risk management"""
        if stop_loss_price <= 0 or entry_price <= 0:
            return 0
        
        # Risk amount in account currency
        risk_amount = self.current_balance * self.max_risk_per_trade
        
        # Price risk in pips (for forex)
        if symbol in ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD']:
            pip_value = 10000  # 4-decimal pip value
        else:
            pip_value = 100    # 2-decimal pip value (JPY pairs)
        
        price_risk_pips = abs(entry_price - stop_loss_price) * pip_value
        
        if price_risk_pips == 0:
            return 0
        
        # Standard lot calculation (100,000 units)
        pip_value_usd = 10 if symbol.endswith('USD') else 10  # Simplified
        max_lots = risk_amount / (price_risk_pips * pip_value_usd)
        
        # Maximum position size (2% of account for conservative approach)
        max_position_value = self.current_balance * 0.02
        max_lots_by_value = max_position_value / (entry_price * 100000)
        
        return min(max_lots, max_lots_by_value, 0.1)  # Max 0.1 lots per trade
    
    def update_floating_pnl(self, current_prices):
        """Update real-time equity with floating P&L"""
        floating_pnl = 0
        
        for symbol, position in self.positions.items():
            if symbol not in current_prices:
                continue
                
            current_price = current_prices[symbol]
            
            if position['type'] == 'long':
                pnl = (current_price - position['entry_price']) * position['size'] * 100000
            else:
                pnl = (position['entry_price'] - current_price) * position['size'] * 100000
            
            floating_pnl += pnl
        
        self.equity = self.current_balance + floating_pnl
        
        # Update peak equity
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
    
    def check_ftmo_violations(self, timestamp):
        """Check for FTMO rule violations in real-time"""
        current_date = timestamp.date()
        
        # Reset daily metrics on new day
        if self.current_date != current_date:
            if self.current_date is not None:
                # Store previous day metrics
                prev_date = self.current_date
                daily_pnl = self.current_balance - self.daily_starting_balance
                self.daily_metrics[prev_date] = {
                    'starting_balance': self.daily_starting_balance,
                    'ending_balance': self.current_balance,
                    'daily_pnl': daily_pnl,
                    'daily_pnl_pct': daily_pnl / self.initial_balance,
                    'trades_count': self.daily_trades_count,
                    'equity_low': min(self.daily_metrics.get(prev_date, {}).get('equity_low', self.equity), self.equity),
                    'equity_high': max(self.daily_metrics.get(prev_date, {}).get('equity_high', self.equity), self.equity)
                }
                
                # Check daily loss violation
                if daily_pnl < -self.initial_balance * self.max_daily_loss:
                    self.challenge_failed = True
                    self.failure_reason = f"Daily loss limit exceeded on {prev_date}: {daily_pnl/self.initial_balance:.2%}"
                    return False
            
            self.current_date = current_date
            self.daily_starting_balance = self.current_balance
            self.daily_trades_count = 0
        
        # Update current day equity extremes
        if current_date not in self.daily_metrics:
            self.daily_metrics[current_date] = {
                'equity_low': self.equity,
                'equity_high': self.equity
            }
        else:
            self.daily_metrics[current_date]['equity_low'] = min(
                self.daily_metrics[current_date]['equity_low'], self.equity
            )
            self.daily_metrics[current_date]['equity_high'] = max(
                self.daily_metrics[current_date]['equity_high'], self.equity
            )
        
        # Check total loss violation (using current equity)
        total_loss = self.initial_balance - self.equity
        if total_loss >= self.initial_balance * self.max_total_loss:
            self.challenge_failed = True
            self.failure_reason = f"Total loss limit exceeded: {total_loss/self.initial_balance:.2%}"
            return False
        
        # Check if daily loss would be exceeded (using equity)
        daily_loss = self.daily_starting_balance - self.equity
        if daily_loss >= self.initial_balance * self.max_daily_loss:
            self.challenge_failed = True
            self.failure_reason = f"Daily loss limit exceeded: {daily_loss/self.initial_balance:.2%}"
            return False
        
        return True
    
    def strategy_london_breakout(self, data, symbol):
        """London Session Breakout Strategy - Professional Implementation"""
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        signals['stop_loss'] = np.nan
        signals['take_profit'] = np.nan
        
        # Parameters
        lookback_hours = 4  # 4-hour range for breakout
        breakout_buffer = 0.0002  # 2 pips buffer
        risk_reward = 2.0  # 1:2 RR ratio
        
        # Calculate 4-hour highs and lows
        high_4h = data['high'].rolling(window=240, min_periods=120).max()  # 4h = 240 minutes
        low_4h = data['low'].rolling(window=240, min_periods=120).min()
        
        # ATR for dynamic stops
        data['hl'] = data['high'] - data['low']
        data['hc'] = np.abs(data['high'] - data['close'].shift())
        data['lc'] = np.abs(data['low'] - data['close'].shift())
        data['tr'] = data[['hl', 'hc', 'lc']].max(axis=1)
        atr = data['tr'].rolling(window=14).mean()
        
        # Only trade during London session (8:00-12:00 GMT)
        london_session = (data['hour'] >= 8) & (data['hour'] < 12)
        
        # Breakout conditions
        long_breakout = (
            london_session &
            (data['close'] > high_4h.shift(1) + breakout_buffer) &
            (data['volume'] > data['volume'].rolling(20).mean())  # Volume confirmation
        )
        
        short_breakout = (
            london_session &
            (data['close'] < low_4h.shift(1) - breakout_buffer) &
            (data['volume'] > data['volume'].rolling(20).mean())
        )
        
        # Set signals
        signals.loc[long_breakout, 'signal'] = 1
        signals.loc[short_breakout, 'signal'] = -1
        
        # Calculate stops and targets
        signals.loc[long_breakout, 'stop_loss'] = data.loc[long_breakout, 'close'] - (atr.loc[long_breakout] * 1.5)
        signals.loc[long_breakout, 'take_profit'] = data.loc[long_breakout, 'close'] + (atr.loc[long_breakout] * 3.0)
        
        signals.loc[short_breakout, 'stop_loss'] = data.loc[short_breakout, 'close'] + (atr.loc[short_breakout] * 1.5)
        signals.loc[short_breakout, 'take_profit'] = data.loc[short_breakout, 'close'] - (atr.loc[short_breakout] * 3.0)
        
        return signals
    
    def strategy_ny_session_momentum(self, data, symbol):
        """New York Session Momentum Strategy"""
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        signals['stop_loss'] = np.nan
        signals['take_profit'] = np.nan
        
        # Calculate indicators
        ema_fast = data['close'].ewm(span=12).mean()
        ema_slow = data['close'].ewm(span=26).mean()
        
        # MACD
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=9).mean()
        macd_histogram = macd - macd_signal
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # NY session (13:00-17:00 GMT)
        ny_session = (data['hour'] >= 13) & (data['hour'] < 17)
        
        # Momentum conditions
        bullish_momentum = (
            ny_session &
            (ema_fast > ema_slow) &
            (macd > macd_signal) &
            (macd_histogram > macd_histogram.shift(1)) &  # Increasing momentum
            (rsi > 50) & (rsi < 70)  # Not overbought
        )
        
        bearish_momentum = (
            ny_session &
            (ema_fast < ema_slow) &
            (macd < macd_signal) &
            (macd_histogram < macd_histogram.shift(1)) &  # Decreasing momentum
            (rsi < 50) & (rsi > 30)  # Not oversold
        )
        
        # Set signals
        signals.loc[bullish_momentum, 'signal'] = 1
        signals.loc[bearish_momentum, 'signal'] = -1
        
        # Dynamic stops based on ATR
        data['hl'] = data['high'] - data['low']
        data['hc'] = np.abs(data['high'] - data['close'].shift())
        data['lc'] = np.abs(data['low'] - data['close'].shift())
        data['tr'] = data[['hl', 'hc', 'lc']].max(axis=1)
        atr = data['tr'].rolling(window=14).mean()
        
        # Calculate stops and targets
        signals.loc[bullish_momentum, 'stop_loss'] = data.loc[bullish_momentum, 'close'] - (atr.loc[bullish_momentum] * 2.0)
        signals.loc[bullish_momentum, 'take_profit'] = data.loc[bullish_momentum, 'close'] + (atr.loc[bullish_momentum] * 3.0)
        
        signals.loc[bearish_momentum, 'stop_loss'] = data.loc[bearish_momentum, 'close'] + (atr.loc[bearish_momentum] * 2.0)
        signals.loc[bearish_momentum, 'take_profit'] = data.loc[bearish_momentum, 'close'] - (atr.loc[bearish_momentum] * 3.0)
        
        return signals
    
    def strategy_mean_reversion_scalp(self, data, symbol):
        """Conservative Mean Reversion Scalping Strategy"""
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        signals['stop_loss'] = np.nan
        signals['take_profit'] = np.nan
        
        # Bollinger Bands
        sma_20 = data['close'].rolling(window=20).mean()
        bb_std = data['close'].rolling(window=20).std()
        bb_upper = sma_20 + (bb_std * 2)
        bb_lower = sma_20 - (bb_std * 2)
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Only during high liquidity sessions
        high_liquidity = (
            ((data['hour'] >= 8) & (data['hour'] < 12)) |  # London
            ((data['hour'] >= 13) & (data['hour'] < 17))   # NY
        )
        
        # Mean reversion conditions
        oversold_bounce = (
            high_liquidity &
            (data['close'] <= bb_lower) &
            (rsi < 30) &
            (data['close'] > data['close'].shift(1))  # Price starting to bounce
        )
        
        overbought_drop = (
            high_liquidity &
            (data['close'] >= bb_upper) &
            (rsi > 70) &
            (data['close'] < data['close'].shift(1))  # Price starting to drop
        )
        
        # Set signals
        signals.loc[oversold_bounce, 'signal'] = 1
        signals.loc[overbought_drop, 'signal'] = -1
        
        # Tight stops for scalping
        pip_value = 0.0001 if symbol in ['EURUSD', 'GBPUSD'] else 0.01
        
        signals.loc[oversold_bounce, 'stop_loss'] = data.loc[oversold_bounce, 'close'] - (10 * pip_value)
        signals.loc[oversold_bounce, 'take_profit'] = data.loc[oversold_bounce, 'close'] + (15 * pip_value)
        
        signals.loc[overbought_drop, 'stop_loss'] = data.loc[overbought_drop, 'close'] + (10 * pip_value)
        signals.loc[overbought_drop, 'take_profit'] = data.loc[overbought_drop, 'close'] - (15 * pip_value)
        
        return signals
    
    def backtest_forex_strategy(self, symbol, strategy_func, strategy_name, start_date, end_date, data_limit=None):
        """Run comprehensive forex backtest with FTMO compliance"""
        print(f"\n🔄 Backtesting {strategy_name} on {symbol}")
        print("=" * 60)
        
        # Reset state
        self.current_balance = self.initial_balance
        self.equity = self.initial_balance
        self.peak_equity = self.initial_balance
        self.trades = []
        self.positions = {}
        self.daily_metrics = {}
        self.trading_days = 0
        self.current_date = None
        self.daily_trades_count = 0
        self.challenge_failed = False
        self.failure_reason = ""
        
        # Fetch data
        print(f"📊 Fetching {symbol} data from {start_date} to {end_date}...")
        data = self.get_forex_data(symbol, start_date, end_date, data_limit)
        
        if data.empty:
            return {'error': f'No data available for {symbol}'}
        
        print(f"✅ Loaded {len(data):,} 1-minute candles")
        print(f"📅 Period: {data.index.min()} to {data.index.max()}")
        
        # Generate trading signals
        print(f"🧠 Generating {strategy_name} signals...")
        signals = strategy_func(data, symbol)
        
        # Track performance metrics
        equity_curve = []
        trade_count = 0
        
        # Process each minute
        for i, (timestamp, row) in enumerate(data.iterrows()):
            if i < 100:  # Skip initial warm-up period
                continue
            
            current_prices = {symbol: row['close']}
            
            # Update floating P&L
            self.update_floating_pnl(current_prices)
            
            # Check FTMO violations
            if not self.check_ftmo_violations(timestamp):
                print(f"🛑 FTMO Challenge Failed: {self.failure_reason}")
                break
            
            # Process existing positions
            positions_to_close = []
            for pos_symbol, position in self.positions.items():
                if pos_symbol == symbol:
                    current_price = row['close']
                    
                    # Check stop loss
                    if ((position['type'] == 'long' and current_price <= position['stop_loss']) or
                        (position['type'] == 'short' and current_price >= position['stop_loss'])):
                        positions_to_close.append((pos_symbol, current_price, 'stop'))
                    
                    # Check take profit
                    elif ((position['type'] == 'long' and current_price >= position['take_profit']) or
                          (position['type'] == 'short' and current_price <= position['take_profit'])):
                        positions_to_close.append((pos_symbol, current_price, 'target'))
            
            # Close positions
            for pos_symbol, exit_price, exit_reason in positions_to_close:
                self.close_position(pos_symbol, exit_price, timestamp, exit_reason)
                trade_count += 1
            
            # Look for new entries
            if (len(self.positions) < self.max_concurrent_positions and
                self.daily_trades_count < self.max_daily_trades and
                self.is_trading_session(timestamp)):
                
                if i in signals.index and signals.loc[timestamp, 'signal'] != 0:
                    signal_type = 'long' if signals.loc[timestamp, 'signal'] > 0 else 'short'
                    entry_price = row['close']
                    stop_loss = signals.loc[timestamp, 'stop_loss']
                    take_profit = signals.loc[timestamp, 'take_profit']
                    
                    if not pd.isna(stop_loss) and not pd.isna(take_profit):
                        if self.open_position(symbol, signal_type, entry_price, stop_loss, take_profit, timestamp):
                            trade_count += 1
            
            # Record equity curve (every 60 minutes to reduce data)
            if i % 60 == 0:
                equity_curve.append({
                    'timestamp': timestamp,
                    'balance': self.current_balance,
                    'equity': self.equity,
                    'drawdown': (self.peak_equity - self.equity) / self.peak_equity,
                    'positions': len(self.positions)
                })
        
        # Close any remaining positions
        final_price = data.iloc[-1]['close']
        for symbol_pos in list(self.positions.keys()):
            self.close_position(symbol_pos, final_price, data.index[-1], 'eod')
        
        return self.calculate_ftmo_results(strategy_name, symbol, equity_curve)
    
    def open_position(self, symbol, position_type, entry_price, stop_loss, take_profit, timestamp):
        """Open a new forex position"""
        position_size = self.calculate_position_size(entry_price, stop_loss, symbol)
        
        if position_size <= 0:
            return False
        
        # Check risk limits
        if len(self.positions) >= self.max_concurrent_positions:
            return False
        
        if self.daily_trades_count >= self.max_daily_trades:
            return False
        
        position_value = position_size * 100000 * entry_price  # Standard lot value
        
        self.positions[symbol] = {
            'type': position_type,
            'size': position_size,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': timestamp,
            'entry_value': position_value
        }
        
        self.daily_trades_count += 1
        
        print(f"📈 {timestamp.strftime('%Y-%m-%d %H:%M')} - Opened {position_type} {symbol}: {position_size:.3f} lots @ {entry_price:.5f}")
        return True
    
    def close_position(self, symbol, exit_price, timestamp, exit_reason):
        """Close a forex position"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        # Calculate P&L in USD
        if position['type'] == 'long':
            pnl_pips = (exit_price - position['entry_price']) * 10000
            pnl_usd = pnl_pips * position['size'] * 10  # $10 per pip per lot
        else:
            pnl_pips = (position['entry_price'] - exit_price) * 10000
            pnl_usd = pnl_pips * position['size'] * 10
        
        # Update balance
        self.current_balance += pnl_usd
        
        # Record trade
        duration_minutes = (timestamp - position['entry_time']).total_seconds() / 60
        
        self.trades.append({
            'symbol': symbol,
            'type': position['type'],
            'entry_time': position['entry_time'],
            'exit_time': timestamp,
            'duration_minutes': duration_minutes,
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'stop_loss': position['stop_loss'],
            'take_profit': position['take_profit'],
            'size_lots': position['size'],
            'pnl_pips': pnl_pips,
            'pnl_usd': pnl_usd,
            'exit_reason': exit_reason
        })
        
        result = "WIN" if pnl_usd > 0 else "LOSS"
        print(f"📉 {timestamp.strftime('%Y-%m-%d %H:%M')} - Closed {position['type']} {symbol}: {pnl_pips:+.1f} pips (${pnl_usd:+.2f}) - {result} ({exit_reason})")
        
        del self.positions[symbol]
    
    def calculate_ftmo_results(self, strategy_name, symbol, equity_curve):
        """Calculate comprehensive FTMO performance metrics"""
        if not self.trades:
            return {
                'strategy': strategy_name,
                'symbol': symbol,
                'ftmo_passed': False,
                'reason': 'No trades executed'
            }
        
        # Basic metrics
        total_return = (self.current_balance - self.initial_balance) / self.initial_balance
        total_trades = len(self.trades)
        
        # Trade analysis
        winning_trades = [t for t in self.trades if t['pnl_usd'] > 0]
        losing_trades = [t for t in self.trades if t['pnl_usd'] <= 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        avg_win = np.mean([t['pnl_usd'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl_usd'] for t in losing_trades]) if losing_trades else 0
        
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # Risk metrics
        equity_values = [point['equity'] for point in equity_curve]
        max_drawdown = 0
        peak = equity_values[0]
        
        for equity in equity_values:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # FTMO Compliance Check
        profit_target_reached = total_return >= self.profit_target
        
        # Check daily loss violations
        daily_violations = []
        for date, metrics in self.daily_metrics.items():
            if 'daily_pnl_pct' in metrics and metrics['daily_pnl_pct'] < -self.max_daily_loss:
                daily_violations.append(date)
        
        no_daily_violations = len(daily_violations) == 0 and not self.challenge_failed
        no_total_loss_violation = total_return > -self.max_total_loss and not self.challenge_failed
        
        # Trading days calculation
        trading_days = len([d for d, m in self.daily_metrics.items() if m.get('trades_count', 0) > 0])
        min_trading_days_met = trading_days >= self.min_trading_days
        
        ftmo_passed = (profit_target_reached and no_daily_violations and 
                      no_total_loss_violation and min_trading_days_met)
        
        # Additional metrics
        avg_trade_duration = np.mean([t['duration_minutes'] for t in self.trades]) if self.trades else 0
        total_pips = sum([t['pnl_pips'] for t in self.trades])
        
        return {
            'strategy': strategy_name,
            'symbol': symbol,
            'total_return_pct': total_return * 100,
            'final_balance': self.current_balance,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown_pct': max_drawdown * 100,
            'avg_trade_duration_min': avg_trade_duration,
            'total_pips': total_pips,
            'trading_days': trading_days,
            'daily_violations': len(daily_violations),
            'ftmo_passed': ftmo_passed,
            'challenge_failed': self.challenge_failed,
            'failure_reason': self.failure_reason,
            'compliance_details': {
                'profit_target_reached': profit_target_reached,
                'no_daily_violations': no_daily_violations,
                'no_total_loss_violation': no_total_loss_violation,
                'min_trading_days_met': min_trading_days_met
            }
        }

def main():
    """Run comprehensive FTMO forex backtesting"""
    print("🏦 FTMO CHALLENGE FOREX BACKTESTING FRAMEWORK")
    print("=" * 80)
    print("Professional Implementation for 30-Year Trading Standards")
    print("=" * 80)
    
    # Initialize backtester
    backtester = FTMOForexBacktester(initial_balance=100000)
    
    # Test parameters
    symbols = ['EURUSD', 'GBPUSD']
    
    # Recent test period (3 months for speed, but comprehensive)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    print(f"💰 Initial Balance: ${backtester.initial_balance:,}")
    print(f"📅 Test Period: {start_date.date()} to {end_date.date()}")
    print(f"🎯 FTMO Target: {backtester.profit_target:.0%} profit")
    print(f"⚠️  Max Daily Loss: {backtester.max_daily_loss:.0%}")
    print(f"🛡️  Max Total Loss: {backtester.max_total_loss:.0%}")
    
    # Test strategies
    strategies = [
        (backtester.strategy_london_breakout, "London Breakout"),
        (backtester.strategy_ny_session_momentum, "NY Momentum"),
        (backtester.strategy_mean_reversion_scalp, "Mean Reversion Scalp")
    ]
    
    all_results = []
    
    # Run tests
    for strategy_func, strategy_name in strategies:
        for symbol in symbols:
            try:
                print(f"\n🔄 Testing {strategy_name} on {symbol}...")
                result = backtester.backtest_forex_strategy(
                    symbol, strategy_func, strategy_name, 
                    start_date, end_date, data_limit=50000  # Limit for speed
                )
                
                if 'error' not in result:
                    all_results.append(result)
                    
            except Exception as e:
                print(f"❌ Error testing {strategy_name} on {symbol}: {e}")
                import traceback
                traceback.print_exc()
    
    # Display results
    print("\n" + "=" * 100)
    print("🏆 FTMO CHALLENGE BACKTEST RESULTS")
    print("=" * 100)
    
    passed_strategies = []
    
    for result in all_results:
        status = "✅ PASSED" if result['ftmo_passed'] else "❌ FAILED"
        
        print(f"\n📊 {result['strategy']} - {result['symbol']} - {status}")
        print("-" * 70)
        print(f"💰 Total Return: {result['total_return_pct']:+.2f}%")
        print(f"💵 Final Balance: ${result['final_balance']:,.2f}")
        print(f"📈 Total Trades: {result['total_trades']}")
        print(f"🎯 Win Rate: {result['win_rate']:.1%}")
        print(f"💡 Profit Factor: {result['profit_factor']:.2f}")
        print(f"📉 Max Drawdown: {result['max_drawdown_pct']:.2f}%")
        print(f"⏱️  Avg Trade Duration: {result['avg_trade_duration_min']:.0f} minutes")
        print(f"📊 Total Pips: {result['total_pips']:+.1f}")
        print(f"🗓️  Trading Days: {result['trading_days']}")
        
        if result['ftmo_passed']:
            passed_strategies.append(result)
        else:
            print(f"❌ Failure Reason: {result.get('failure_reason', 'Did not meet FTMO requirements')}")
            
            compliance = result['compliance_details']
            if not compliance['profit_target_reached']:
                print(f"   • Profit target not reached ({result['total_return_pct']:.1f}% < 10%)")
            if not compliance['no_daily_violations']:
                print(f"   • Daily loss violations: {result['daily_violations']}")
            if not compliance['no_total_loss_violation']:
                print(f"   • Total loss violation")
            if not compliance['min_trading_days_met']:
                print(f"   • Insufficient trading days ({result['trading_days']} < 4)")
    
    # Final recommendations
    print(f"\n" + "=" * 100)
    print("🎯 FTMO CHALLENGE RECOMMENDATIONS")
    print("=" * 100)
    
    if passed_strategies:
        best_strategy = max(passed_strategies, key=lambda x: x['total_return_pct'])
        
        print(f"\n🏆 RECOMMENDED STRATEGY:")
        print(f"Strategy: {best_strategy['strategy']}")
        print(f"Symbol: {best_strategy['symbol']}")
        print(f"Expected Return: {best_strategy['total_return_pct']:+.2f}%")
        print(f"Win Rate: {best_strategy['win_rate']:.1%}")
        print(f"Profit Factor: {best_strategy['profit_factor']:.2f}")
        print(f"Max Drawdown: {best_strategy['max_drawdown_pct']:.2f}%")
        
        print(f"\n📋 IMPLEMENTATION GUIDELINES:")
        print(f"• Risk per trade: 1% maximum")
        print(f"• Maximum concurrent positions: 2")
        print(f"• Daily trade limit: 5")
        print(f"• Trading sessions: London (8-12 GMT) & NY (13-17 GMT)")
        print(f"• Monitor daily P&L closely (stop at -4.5%)")
        print(f"• Expected average trade duration: {best_strategy['avg_trade_duration_min']:.0f} minutes")
        
    else:
        print("\n⚠️  NO STRATEGIES PASSED FTMO REQUIREMENTS")
        print("\n🔧 RECOMMENDED IMPROVEMENTS:")
        print("• Reduce risk per trade to 0.5%")
        print("• Implement tighter entry filters")
        print("• Add maximum daily loss stop at 4% instead of 5%")
        print("• Consider longer timeframe strategies")
        print("• Focus on higher probability setups only")
        
        if all_results:
            best_failed = max(all_results, key=lambda x: x['total_return_pct'])
            print(f"\n📊 Best Non-Passing Strategy:")
            print(f"Strategy: {best_failed['strategy']} - {best_failed['symbol']}")
            print(f"Return: {best_failed['total_return_pct']:+.2f}%")
            print(f"Main Issue: {best_failed.get('failure_reason', 'Multiple compliance issues')}")

if __name__ == "__main__":
    main()