import os
#!/usr/bin/env python3
"""
FTMO Challenge Realistic Backtesting Framework
Works with daily OHLC data and simulates intraday risk management
"""

import pandas as pd
import numpy as np
import psycopg2
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import json
warnings.filterwarnings('ignore')

class FTMORealisticBacktester:
    def __init__(self, initial_balance=100000):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.peak_balance = initial_balance
        self.daily_pnl = {}
        self.trades = []
        self.positions = {}
        self.trading_days = 0
        self.violated_days = []

        # FTMO Rules
        self.profit_target = 0.10  # 10%
        self.max_daily_loss = 0.05  # 5%
        self.max_total_loss = 0.10  # 10%
        self.min_trading_days = 4

        # Conservative Risk Management for Daily Data
        self.max_risk_per_trade = 0.01  # 1% risk per trade (very conservative)
        self.max_concurrent_positions = 2  # Reduce concurrent positions
        self.max_daily_trades = 3  # Limit daily trades
        self.position_holding_days = 5  # Average holding period

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
            return None

    def get_daily_ohlc_data(self, symbols, start_date, end_date):
        """Fetch OHLC daily data"""
        conn = self.connect_to_db()

        if conn:
            try:
                query = """
                SELECT symbol, date, open, high, low, close, volume
                FROM dax_historical_daily
                WHERE symbol IN %s AND date BETWEEN %s AND %s
                ORDER BY date, symbol
                """
                df = pd.read_sql_query(query, conn, params=(tuple(symbols), start_date, end_date))
                conn.close()

                # Reshape data for easier access
                ohlc_data = {}
                for symbol in symbols:
                    symbol_data = df[df['symbol'] == symbol].set_index('date')
                    if not symbol_data.empty:
                        ohlc_data[symbol] = symbol_data[['open', 'high', 'low', 'close', 'volume']]

                return ohlc_data
            except Exception as e:
                print(f"Database error: {e}")
                conn.close()

        # Fallback to yfinance
        print("Fetching OHLC data from yfinance...")
        ohlc_data = {}
        for symbol in symbols:
            try:
                data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                if not data.empty:
                    ohlc_data[symbol] = data[['Open', 'High', 'Low', 'Close', 'Volume']]
                    ohlc_data[symbol].columns = ['open', 'high', 'low', 'close', 'volume']
            except:
                continue

        return ohlc_data

    def simulate_intraday_risk(self, symbol, entry_price, stop_loss, target_price, daily_high, daily_low):
        """
        Simulate intraday price movement and check if stop loss would be hit
        Uses daily high/low to estimate intraday volatility
        """
        daily_range = (daily_high - daily_low) / entry_price

        # Conservative assumption: if daily range suggests stop could be hit
        if entry_price > stop_loss:  # Long position
            stop_risk = (entry_price - stop_loss) / entry_price
            if stop_risk <= daily_range * 0.8:  # 80% chance if within daily range
                return 'stopped_out'
            elif daily_high >= target_price:
                return 'target_hit'
        else:  # Short position
            stop_risk = (stop_loss - entry_price) / entry_price
            if stop_risk <= daily_range * 0.8:
                return 'stopped_out'
            elif daily_low <= target_price:
                return 'target_hit'

        return 'hold'

    def calculate_position_size_conservative(self, entry_price, stop_loss_price):
        """Calculate ultra-conservative position size for daily data"""
        if stop_loss_price <= 0 or entry_price <= 0:
            return 0

        risk_amount = self.current_balance * self.max_risk_per_trade
        price_risk = abs(entry_price - stop_loss_price)

        if price_risk == 0:
            return 0

        position_size = risk_amount / price_risk

        # Maximum position value (20% of account for daily data)
        max_position_value = self.current_balance * 0.20
        max_shares = max_position_value / entry_price

        return min(position_size, max_shares)

    def strategy_conservative_swing(self, ohlc_data, lookback=20):
        """Conservative swing trading strategy for daily data"""
        signals = {}

        for symbol, data in ohlc_data.items():
            if symbol == '^GDAXI' or len(data) < lookback:
                continue

            # Calculate technical indicators
            close = data['close']
            high = data['high']
            low = data['low']

            # Moving averages
            sma_20 = close.rolling(window=20).mean()
            sma_50 = close.rolling(window=50).mean()

            # Bollinger Bands
            bb_std = close.rolling(window=20).std()
            bb_upper = sma_20 + (bb_std * 2)
            bb_lower = sma_20 - (bb_std * 2)

            # RSI
            rsi = self.calculate_rsi(close, 14)

            # ATR for stop loss
            atr = self.calculate_atr(high, low, close, 14)

            # Conservative entry conditions
            long_condition = (
                (close < bb_lower) &  # Oversold
                (rsi < 30) &  # RSI oversold
                (close > sma_50) &  # Above long-term trend
                (close.shift(1) < bb_lower.shift(1))  # First touch of lower band
            )

            short_condition = (
                (close > bb_upper) &  # Overbought
                (rsi > 70) &  # RSI overbought
                (close < sma_50) &  # Below long-term trend
                (close.shift(1) > bb_upper.shift(1))  # First touch of upper band
            )

            # Calculate stops and targets
            stop_multiplier = 2.0  # 2 ATR stop loss
            target_multiplier = 3.0  # 3:1 reward to risk

            signals[symbol] = {
                'long_signals': long_condition,
                'short_signals': short_condition,
                'close_prices': close,
                'atr': atr,
                'sma_20': sma_20,
                'rsi': rsi,
                'high': high,
                'low': low,
                'stop_multiplier': stop_multiplier,
                'target_multiplier': target_multiplier
            }

        return signals

    def strategy_breakout_momentum(self, ohlc_data, lookback=20):
        """Breakout momentum strategy with tight risk control"""
        signals = {}

        for symbol, data in ohlc_data.items():
            if symbol == '^GDAXI' or len(data) < lookback:
                continue

            close = data['close']
            high = data['high']
            low = data['low']
            volume = data['volume']

            # Breakout levels
            high_20 = high.rolling(window=20).max()
            low_20 = low.rolling(window=20).min()

            # Volume confirmation
            avg_volume = volume.rolling(window=20).mean()
            volume_surge = volume > avg_volume * 1.5

            # ATR for stops
            atr = self.calculate_atr(high, low, close, 14)

            # Breakout conditions
            long_breakout = (
                (high > high_20.shift(1)) &  # New 20-day high
                volume_surge &  # Volume confirmation
                (close > close.shift(1) * 1.02)  # Strong close
            )

            short_breakout = (
                (low < low_20.shift(1)) &  # New 20-day low
                volume_surge &  # Volume confirmation
                (close < close.shift(1) * 0.98)  # Weak close
            )

            signals[symbol] = {
                'long_signals': long_breakout,
                'short_signals': short_breakout,
                'close_prices': close,
                'atr': atr,
                'high': high,
                'low': low,
                'stop_multiplier': 1.5,  # Tighter stops for breakouts
                'target_multiplier': 2.5
            }

        return signals

    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_atr(self, high, low, close, period=14):
        """Calculate Average True Range"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(window=period).mean()

    def backtest_daily_strategy(self, ohlc_data, strategy_func, strategy_name):
        """Run backtest with daily data and FTMO simulation"""
        print(f"\nBacktesting Strategy: {strategy_name}")
        print("=" * 50)

        # Reset state
        self.current_balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.daily_pnl = {}
        self.trades = []
        self.positions = {}
        self.trading_days = 0
        self.violated_days = []

        # Generate signals
        signals = strategy_func(ohlc_data)

        # Get all trading dates
        all_dates = set()
        for symbol_data in ohlc_data.values():
            all_dates.update(symbol_data.index)
        trading_dates = sorted(all_dates)

        # Track metrics
        equity_curve = []
        max_drawdown = 0
        daily_trades_count = 0

        # Simulate trading day by day
        for i, date in enumerate(trading_dates[50:], 50):  # Skip warm-up period
            daily_start_balance = self.current_balance
            daily_trades_count = 0

            # Check existing positions first
            positions_to_close = []
            for symbol, position in list(self.positions.items()):
                if symbol not in ohlc_data or date not in ohlc_data[symbol].index:
                    continue

                daily_data = ohlc_data[symbol].loc[date]
                days_held = (date - position['entry_date']).days

                # Simulate intraday movement
                intraday_result = self.simulate_intraday_risk(
                    symbol, position['entry_price'], position['stop_loss'],
                    position['target_price'], daily_data['high'], daily_data['low']
                )

                # Close position based on rules
                if (intraday_result == 'stopped_out' or
                    intraday_result == 'target_hit' or
                    days_held >= self.position_holding_days):

                    if intraday_result == 'stopped_out':
                        exit_price = position['stop_loss']
                    elif intraday_result == 'target_hit':
                        exit_price = position['target_price']
                    else:
                        exit_price = daily_data['close']

                    self.close_position(symbol, exit_price, date)
                    daily_trades_count += 1

            # Look for new entries
            if (len(self.positions) < self.max_concurrent_positions and
                daily_trades_count < self.max_daily_trades):

                for symbol, signal_data in signals.items():
                    if symbol in self.positions or date not in signal_data['close_prices'].index:
                        continue

                    if symbol not in ohlc_data or date not in ohlc_data[symbol].index:
                        continue

                    daily_data = ohlc_data[symbol].loc[date]

                    # Check for long signal
                    if (date in signal_data['long_signals'].index and
                        signal_data['long_signals'].loc[date] and
                        daily_trades_count < self.max_daily_trades):

                        entry_price = daily_data['close']
                        atr_value = signal_data['atr'].loc[date] if date in signal_data['atr'].index else entry_price * 0.02
                        stop_loss = entry_price - (atr_value * signal_data['stop_multiplier'])
                        target_price = entry_price + (atr_value * signal_data['target_multiplier'])

                        if self.open_position(symbol, 'long', entry_price, stop_loss, target_price, date):
                            daily_trades_count += 1

                    # Check for short signal
                    elif (date in signal_data['short_signals'].index and
                          signal_data['short_signals'].loc[date] and
                          daily_trades_count < self.max_daily_trades):

                        entry_price = daily_data['close']
                        atr_value = signal_data['atr'].loc[date] if date in signal_data['atr'].index else entry_price * 0.02
                        stop_loss = entry_price + (atr_value * signal_data['stop_multiplier'])
                        target_price = entry_price - (atr_value * signal_data['target_multiplier'])

                        if self.open_position(symbol, 'short', entry_price, stop_loss, target_price, date):
                            daily_trades_count += 1

            # Calculate daily P&L and check FTMO rules
            daily_pnl = self.current_balance - daily_start_balance
            self.update_daily_pnl(date, daily_pnl)

            # Check FTMO violations
            daily_loss_limit = self.initial_balance * self.max_daily_loss
            if daily_pnl < -daily_loss_limit:
                print(f"⚠️  Daily loss limit exceeded on {date}: €{daily_pnl:.2f}")
                self.violated_days.append(date)
                # In real FTMO, this would stop trading for the day

            # Check total loss limit
            total_loss = self.initial_balance - self.current_balance
            if total_loss >= self.initial_balance * self.max_total_loss:
                print(f"🛑 Total loss limit exceeded on {date}. Challenge failed.")
                break

            # Update trading days
            if daily_trades_count > 0 or daily_pnl != 0:
                self.trading_days += 1

            # Update metrics
            equity_curve.append({
                'date': date,
                'balance': self.current_balance,
                'daily_pnl': daily_pnl,
                'trades_today': daily_trades_count
            })

            # Update peak and drawdown
            if self.current_balance > self.peak_balance:
                self.peak_balance = self.current_balance

            drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
            max_drawdown = max(max_drawdown, drawdown)

        return self.calculate_ftmo_performance_metrics(equity_curve, max_drawdown, strategy_name)

    def open_position(self, symbol, position_type, entry_price, stop_loss, target_price, date):
        """Open a new position"""
        position_size = self.calculate_position_size_conservative(entry_price, stop_loss)

        if position_size <= 0:
            return False

        position_value = position_size * entry_price

        # Check if we have enough balance (conservative 20% max per position)
        if position_value > self.current_balance * 0.20:
            return False

        self.positions[symbol] = {
            'type': position_type,
            'size': position_size,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'target_price': target_price,
            'entry_date': date,
            'entry_value': position_value
        }

        print(f"📈 Opened {position_type} position in {symbol}: {position_size:.0f} shares at €{entry_price:.2f} (Stop: €{stop_loss:.2f}, Target: €{target_price:.2f})")
        return True

    def close_position(self, symbol, exit_price, date):
        """Close an existing position"""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]

        if position['type'] == 'long':
            pnl = (exit_price - position['entry_price']) * position['size']
        else:
            pnl = (position['entry_price'] - exit_price) * position['size']

        self.current_balance += pnl

        # Record trade
        holding_days = (date - position['entry_date']).days
        return_pct = pnl / position['entry_value']

        self.trades.append({
            'symbol': symbol,
            'type': position['type'],
            'entry_date': position['entry_date'],
            'exit_date': date,
            'holding_days': holding_days,
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'stop_loss': position['stop_loss'],
            'target_price': position['target_price'],
            'size': position['size'],
            'pnl': pnl,
            'return_pct': return_pct
        })

        result = "WIN" if pnl > 0 else "LOSS"
        print(f"📉 Closed {position['type']} position in {symbol}: €{pnl:.2f} ({return_pct:.1%}) - {result}")

        del self.positions[symbol]

    def update_daily_pnl(self, date, pnl):
        """Update daily P&L tracking"""
        date_key = date.date() if hasattr(date, 'date') else date
        if date_key not in self.daily_pnl:
            self.daily_pnl[date_key] = 0
        self.daily_pnl[date_key] += pnl

    def calculate_ftmo_performance_metrics(self, equity_curve, max_drawdown, strategy_name):
        """Calculate comprehensive FTMO performance metrics"""
        total_return = (self.current_balance - self.initial_balance) / self.initial_balance
        total_trades = len(self.trades)

        if total_trades == 0:
            return {
                'strategy': strategy_name,
                'total_return': 0,
                'ftmo_compliant': False,
                'reason': 'No trades executed'
            }

        # Trade statistics
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        losing_trades = [t for t in self.trades if t['pnl'] <= 0]

        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

        # Average holding period
        avg_holding_days = np.mean([t['holding_days'] for t in self.trades]) if self.trades else 0

        # Risk metrics
        daily_returns = []
        for i in range(1, len(equity_curve)):
            daily_ret = (equity_curve[i]['balance'] - equity_curve[i-1]['balance']) / equity_curve[i-1]['balance']
            daily_returns.append(daily_ret)

        sharpe_ratio = 0
        if len(daily_returns) > 1:
            mean_return = np.mean(daily_returns)
            std_return = np.std(daily_returns)
            if std_return > 0:
                sharpe_ratio = mean_return / std_return * np.sqrt(252)

        # FTMO Compliance Check
        profit_target_reached = total_return >= self.profit_target
        no_daily_loss_violations = len(self.violated_days) == 0
        no_total_loss_violation = total_return >= -self.max_total_loss
        min_trading_days_met = self.trading_days >= self.min_trading_days

        ftmo_compliant = (profit_target_reached and no_daily_loss_violations and
                         no_total_loss_violation and min_trading_days_met)

        # Calculate maximum daily loss
        max_daily_loss_pct = 0
        if self.daily_pnl:
            max_daily_loss = min(self.daily_pnl.values())
            max_daily_loss_pct = max_daily_loss / self.initial_balance

        return {
            'strategy': strategy_name,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'final_balance': self.current_balance,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_holding_days': avg_holding_days,
            'max_drawdown': max_drawdown,
            'max_daily_loss_pct': max_daily_loss_pct,
            'sharpe_ratio': sharpe_ratio,
            'trading_days': self.trading_days,
            'violated_days_count': len(self.violated_days),
            'ftmo_compliant': ftmo_compliant,
            'compliance_details': {
                'profit_target_reached': profit_target_reached,
                'no_daily_loss_violations': no_daily_loss_violations,
                'no_total_loss_violation': no_total_loss_violation,
                'min_trading_days_met': min_trading_days_met
            }
        }

def main():
    """Run realistic FTMO backtesting with daily data"""
    print("FTMO Challenge Realistic Backtesting (Daily Data)")
    print("=" * 60)

    # Initialize backtester
    backtester = FTMORealisticBacktester(initial_balance=100000)

    # Focus on most liquid DAX stocks
    symbols = ['^GDAXI', 'SAP.DE', 'SIE.DE', 'ALV.DE', 'BMW.DE', 'BAS.DE', 'ADS.DE']

    # Test period - last 1 year for realistic testing
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=365)

    print(f"Testing period: {start_date} to {end_date}")
    print(f"Initial balance: €{backtester.initial_balance:,}")
    print(f"Symbols: {symbols}")

    # Fetch OHLC data
    print("\nFetching OHLC market data...")
    ohlc_data = backtester.get_daily_ohlc_data(symbols, start_date, end_date)

    if not ohlc_data:
        print("No market data available. Exiting.")
        return

    print(f"Data available for {len(ohlc_data)} symbols")
    for symbol, data in ohlc_data.items():
        print(f"  {symbol}: {len(data)} days ({data.index.min()} to {data.index.max()})")

    # Test strategies
    strategies = [
        (backtester.strategy_conservative_swing, "Conservative Swing Trading"),
        (backtester.strategy_breakout_momentum, "Breakout Momentum")
    ]

    results = []

    for strategy_func, strategy_name in strategies:
        try:
            result = backtester.backtest_daily_strategy(ohlc_data, strategy_func, strategy_name)
            results.append(result)
        except Exception as e:
            print(f"Error testing {strategy_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Display comprehensive results
    print("\n" + "=" * 80)
    print("FTMO CHALLENGE REALISTIC BACKTEST RESULTS")
    print("=" * 80)

    for result in results:
        print(f"\n📊 Strategy: {result['strategy']}")
        print("-" * 50)
        print(f"💰 Total Return: {result['total_return_pct']:.2f}%")
        print(f"💵 Final Balance: €{result['final_balance']:,.2f}")
        print(f"📈 Total Trades: {result['total_trades']}")
        print(f"🎯 Win Rate: {result['win_rate']:.1%}")
        print(f"💡 Profit Factor: {result['profit_factor']:.2f}")
        print(f"📉 Max Drawdown: {result['max_drawdown']:.1%}")
        print(f"📅 Avg Holding: {result['avg_holding_days']:.1f} days")
        print(f"📊 Sharpe Ratio: {result['sharpe_ratio']:.2f}")
        print(f"🗓️  Trading Days: {result['trading_days']}")
        print(f"⚠️  Daily Violations: {result['violated_days_count']}")
        print(f"📉 Max Daily Loss: {result['max_daily_loss_pct']:.1%}")

        # FTMO Compliance
        print(f"\n🏆 FTMO COMPLIANCE: {'✅ PASS' if result['ftmo_compliant'] else '❌ FAIL'}")

        compliance = result['compliance_details']
        print(f"  📈 Profit Target (10%): {'✅' if compliance['profit_target_reached'] else '❌'} ({result['total_return_pct']:.1f}%)")
        print(f"  🚫 No Daily Loss Violations: {'✅' if compliance['no_daily_loss_violations'] else '❌'} ({result['violated_days_count']} violations)")
        print(f"  🛡️  No Total Loss Violations: {'✅' if compliance['no_total_loss_violation'] else '❌'}")
        print(f"  📅 Min Trading Days (4): {'✅' if compliance['min_trading_days_met'] else '❌'} ({result['trading_days']} days)")

    # Recommendations
    print(f"\n" + "=" * 80)
    print("RECOMMENDATIONS FOR FTMO CHALLENGE")
    print("=" * 80)

    ftmo_compliant_strategies = [r for r in results if r['ftmo_compliant']]

    if ftmo_compliant_strategies:
        best_strategy = max(ftmo_compliant_strategies, key=lambda x: x['total_return'])
        print(f"\n🏆 RECOMMENDED STRATEGY: {best_strategy['strategy']}")
        print(f"✅ Expected Return: {best_strategy['total_return_pct']:.2f}%")
        print(f"✅ Win Rate: {best_strategy['win_rate']:.1%}")
        print(f"✅ Max Drawdown: {best_strategy['max_drawdown']:.1%}")
        print(f"✅ Trading Days: {best_strategy['trading_days']}")

        print(f"\n🎯 IMPLEMENTATION TIPS:")
        print(f"• Use 1% risk per trade maximum")
        print(f"• Limit to 2 concurrent positions")
        print(f"• Average holding period: {best_strategy['avg_holding_days']:.1f} days")
        print(f"• Focus on high-probability setups only")
        print(f"• Monitor daily P&L closely (max -5%)")

    else:
        print("\n⚠️  NO STRATEGIES MEET FTMO REQUIREMENTS")
        print("\n🔧 SUGGESTED IMPROVEMENTS:")
        print("• Reduce risk per trade to 0.5%")
        print("• Increase position holding time")
        print("• Add more conservative entry filters")
        print("• Implement tighter daily loss monitoring")

        if results:
            best_non_compliant = max(results, key=lambda x: x['total_return'])
            print(f"\n📊 Best Non-Compliant Strategy: {best_non_compliant['strategy']}")
            print(f"Return: {best_non_compliant['total_return_pct']:.2f}%")
            print(f"Issue: {best_non_compliant['violated_days_count']} daily violations")

if __name__ == "__main__":
    main()