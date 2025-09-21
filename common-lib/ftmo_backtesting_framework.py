import os
#!/usr/bin/env python3
"""
FTMO Challenge Compliant Trading Strategy Backtesting Framework
Implements multiple strategies with strict FTMO risk management rules
"""

import pandas as pd
import numpy as np
import psycopg2
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FTMOBacktester:
    def __init__(self, initial_balance=100000):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.peak_balance = initial_balance
        self.daily_pnl = {}
        self.trades = []
        self.positions = {}
        self.trading_days = 0
        
        # FTMO Rules
        self.profit_target = 0.10  # 10%
        self.max_daily_loss = 0.05  # 5%
        self.max_total_loss = 0.10  # 10%
        self.min_trading_days = 4
        
        # Risk Management Parameters
        self.max_risk_per_trade = 0.015  # 1.5% risk per trade (conservative)
        self.max_concurrent_positions = 3
        self.max_sector_exposure = 0.35  # 35% max in any sector
        self.max_correlation_exposure = 0.50  # 50% max in correlated assets
        
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
            # Fallback for local testing
            print("Using yfinance for data...")
            return None
    
    def get_market_data(self, symbols, start_date, end_date):
        """Fetch market data from database or yfinance"""
        conn = self.connect_to_db()
        
        if conn:
            try:
                query = """
                SELECT symbol, date, open, high, low, close, volume
                FROM dax_historical_daily 
                WHERE symbol IN %s AND date BETWEEN %s AND %s
                ORDER BY symbol, date
                """
                df = pd.read_sql_query(query, conn, params=(tuple(symbols), start_date, end_date))
                conn.close()
                return df.pivot_table(index='date', columns='symbol', values='close')
            except:
                conn.close()
        
        # Fallback to yfinance
        print("Fetching data from yfinance...")
        data = yf.download(symbols, start=start_date, end=end_date)['Close']
        return data.fillna(method='ffill')
    
    def calculate_position_size(self, price, stop_loss_price, symbol):
        """Calculate position size based on FTMO risk rules"""
        if stop_loss_price <= 0 or price <= 0:
            return 0
            
        risk_amount = self.current_balance * self.max_risk_per_trade
        price_risk = abs(price - stop_loss_price)
        
        if price_risk == 0:
            return 0
            
        position_size = risk_amount / price_risk
        
        # Maximum position value (35% of account)
        max_position_value = self.current_balance * 0.35
        max_shares = max_position_value / price
        
        return min(position_size, max_shares)
    
    def check_daily_loss_limit(self, current_date):
        """Check if daily loss limit would be exceeded"""
        today_pnl = self.daily_pnl.get(current_date, 0)
        max_daily_loss_amount = self.initial_balance * self.max_daily_loss
        
        return today_pnl <= -max_daily_loss_amount
    
    def check_total_loss_limit(self):
        """Check if total loss limit would be exceeded"""
        total_loss = self.initial_balance - self.current_balance
        max_total_loss_amount = self.initial_balance * self.max_total_loss
        
        return total_loss >= max_total_loss_amount
    
    def update_daily_pnl(self, current_date, pnl):
        """Update daily P&L tracking"""
        if current_date not in self.daily_pnl:
            self.daily_pnl[current_date] = 0
        self.daily_pnl[current_date] += pnl
    
    def strategy_mean_reversion_conservative(self, data, lookback=20):
        """Conservative mean reversion strategy for FTMO"""
        signals = pd.DataFrame(index=data.index)
        
        for symbol in data.columns:
            if symbol == '^GDAXI':
                continue
                
            prices = data[symbol].dropna()
            if len(prices) < lookback:
                continue
                
            # Calculate indicators
            sma = prices.rolling(window=lookback).mean()
            std = prices.rolling(window=lookback).std()
            rsi = self.calculate_rsi(prices, 14)
            
            # Conservative entry conditions
            oversold = (prices < (sma - 1.5 * std)) & (rsi < 35)
            overbought = (prices > (sma + 1.5 * std)) & (rsi > 65)
            
            signals[f'{symbol}_long'] = oversold
            signals[f'{symbol}_short'] = overbought
            signals[f'{symbol}_price'] = prices
            signals[f'{symbol}_sma'] = sma
            signals[f'{symbol}_stop'] = np.where(oversold, prices * 0.97, 
                                               np.where(overbought, prices * 1.03, np.nan))
            
        return signals
    
    def strategy_momentum_breakout(self, data, lookback=10):
        """Momentum breakout strategy with tight risk control"""
        signals = pd.DataFrame(index=data.index)
        
        for symbol in data.columns:
            if symbol == '^GDAXI':
                continue
                
            prices = data[symbol].dropna()
            if len(prices) < lookback:
                continue
                
            # Calculate indicators
            high_20 = prices.rolling(window=20).max()
            low_20 = prices.rolling(window=20).min()
            volume_sma = data.get(f'{symbol}_volume', pd.Series(index=prices.index, data=1)).rolling(10).mean()
            
            # Breakout conditions
            breakout_long = (prices > high_20.shift(1)) & (prices > prices.shift(1) * 1.005)
            breakout_short = (prices < low_20.shift(1)) & (prices < prices.shift(1) * 0.995)
            
            signals[f'{symbol}_long'] = breakout_long
            signals[f'{symbol}_short'] = breakout_short
            signals[f'{symbol}_price'] = prices
            signals[f'{symbol}_stop'] = np.where(breakout_long, prices * 0.98, 
                                               np.where(breakout_short, prices * 1.02, np.nan))
            
        return signals
    
    def strategy_trend_following(self, data):
        """Trend following with multiple timeframe confirmation"""
        signals = pd.DataFrame(index=data.index)
        
        for symbol in data.columns:
            if symbol == '^GDAXI':
                continue
                
            prices = data[symbol].dropna()
            if len(prices) < 50:
                continue
                
            # Multiple EMAs for trend confirmation
            ema_fast = prices.ewm(span=12).mean()
            ema_slow = prices.ewm(span=26).mean()
            ema_filter = prices.ewm(span=50).mean()
            
            # MACD
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=9).mean()
            
            # Trend conditions
            bullish_trend = (ema_fast > ema_slow) & (prices > ema_filter) & (macd > macd_signal)
            bearish_trend = (ema_fast < ema_slow) & (prices < ema_filter) & (macd < macd_signal)
            
            signals[f'{symbol}_long'] = bullish_trend & (bullish_trend.shift(1) == False)
            signals[f'{symbol}_short'] = bearish_trend & (bearish_trend.shift(1) == False)
            signals[f'{symbol}_price'] = prices
            signals[f'{symbol}_stop'] = np.where(bullish_trend, ema_slow * 0.98, 
                                               np.where(bearish_trend, ema_slow * 1.02, np.nan))
            
        return signals
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def backtest_strategy(self, data, strategy_func, strategy_name):
        """Run backtest with FTMO compliance checks"""
        print(f"\nBacktesting Strategy: {strategy_name}")
        print("=" * 50)
        
        # Reset state
        self.current_balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.daily_pnl = {}
        self.trades = []
        self.positions = {}
        self.trading_days = 0
        
        # Generate signals
        signals = strategy_func(data)
        
        # Track metrics
        daily_balances = []
        equity_curve = []
        max_drawdown = 0
        consecutive_losses = 0
        max_consecutive_losses = 0
        
        # Simulate trading
        for i, date in enumerate(data.index[50:], 50):  # Start after indicator warm-up
            current_date = date.date() if hasattr(date, 'date') else date
            daily_start_balance = self.current_balance
            
            # Check if we should stop trading (FTMO violations)
            if self.check_total_loss_limit():
                print(f"STOPPED: Total loss limit exceeded on {current_date}")
                break
                
            if self.check_daily_loss_limit(current_date):
                print(f"STOPPED: Daily loss limit exceeded on {current_date}")
                continue
            
            # Process existing positions first
            positions_to_close = []
            for symbol, position in self.positions.items():
                current_price = data.loc[date, symbol] if symbol in data.columns else None
                if current_price is None or pd.isna(current_price):
                    continue
                    
                # Check stop loss
                if position['type'] == 'long' and current_price <= position['stop_loss']:
                    positions_to_close.append(symbol)
                elif position['type'] == 'short' and current_price >= position['stop_loss']:
                    positions_to_close.append(symbol)
                # Check profit target (3R profit)
                elif position['type'] == 'long' and current_price >= position['entry_price'] * 1.045:
                    positions_to_close.append(symbol)
                elif position['type'] == 'short' and current_price <= position['entry_price'] * 0.955:
                    positions_to_close.append(symbol)
            
            # Close positions
            for symbol in positions_to_close:
                self.close_position(symbol, data.loc[date, symbol], current_date)
            
            # Look for new entries
            if len(self.positions) < self.max_concurrent_positions:
                for symbol in data.columns:
                    if symbol == '^GDAXI' or symbol in self.positions:
                        continue
                        
                    current_price = data.loc[date, symbol]
                    if pd.isna(current_price):
                        continue
                    
                    # Check for long signal
                    long_signal_col = f'{symbol}_long'
                    if long_signal_col in signals.columns and signals.loc[date, long_signal_col]:
                        stop_price = signals.loc[date, f'{symbol}_stop'] if f'{symbol}_stop' in signals.columns else current_price * 0.97
                        self.open_position(symbol, 'long', current_price, stop_price, current_date)
                    
                    # Check for short signal
                    short_signal_col = f'{symbol}_short'
                    if short_signal_col in signals.columns and signals.loc[date, short_signal_col]:
                        stop_price = signals.loc[date, f'{symbol}_stop'] if f'{symbol}_stop' in signals.columns else current_price * 1.03
                        self.open_position(symbol, 'short', current_price, stop_price, current_date)
            
            # Update daily P&L
            daily_pnl = self.current_balance - daily_start_balance
            self.update_daily_pnl(current_date, daily_pnl)
            
            # Track trading days
            if daily_pnl != 0:
                self.trading_days += 1
            
            # Update metrics
            daily_balances.append(self.current_balance)
            equity_curve.append({
                'date': current_date,
                'balance': self.current_balance,
                'drawdown': (self.peak_balance - self.current_balance) / self.peak_balance
            })
            
            # Update peak balance
            if self.current_balance > self.peak_balance:
                self.peak_balance = self.current_balance
                consecutive_losses = 0
            else:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            
            # Calculate max drawdown
            drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate final metrics
        return self.calculate_performance_metrics(daily_balances, equity_curve, max_drawdown, strategy_name)
    
    def open_position(self, symbol, position_type, entry_price, stop_price, date):
        """Open a new position"""
        position_size = self.calculate_position_size(entry_price, stop_price, symbol)
        
        if position_size <= 0:
            return
        
        position_value = position_size * entry_price
        
        # Check if we have enough balance
        if position_value > self.current_balance * 0.35:  # Max 35% per position
            return
        
        self.positions[symbol] = {
            'type': position_type,
            'size': position_size,
            'entry_price': entry_price,
            'stop_loss': stop_price,
            'entry_date': date,
            'entry_value': position_value
        }
        
        print(f"Opened {position_type} position in {symbol}: {position_size:.0f} shares at €{entry_price:.2f}")
    
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
        self.trades.append({
            'symbol': symbol,
            'type': position['type'],
            'entry_date': position['entry_date'],
            'exit_date': date,
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'size': position['size'],
            'pnl': pnl,
            'return_pct': pnl / position['entry_value']
        })
        
        result = "WIN" if pnl > 0 else "LOSS"
        print(f"Closed {position['type']} position in {symbol}: €{pnl:.2f} ({result})")
        
        del self.positions[symbol]
    
    def calculate_performance_metrics(self, daily_balances, equity_curve, max_drawdown, strategy_name):
        """Calculate comprehensive performance metrics"""
        total_return = (self.current_balance - self.initial_balance) / self.initial_balance
        total_trades = len(self.trades)
        
        if total_trades == 0:
            return {
                'strategy': strategy_name,
                'total_return': 0,
                'ftmo_compliant': False,
                'reason': 'No trades executed'
            }
        
        # Win rate
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # Average win/loss
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        losing_trades = [t for t in self.trades if t['pnl'] <= 0]
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        # Sharpe ratio (simplified)
        daily_returns = np.diff(daily_balances) / daily_balances[:-1]
        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 1 else 0
        
        # Check FTMO compliance
        profit_target_reached = total_return >= self.profit_target
        no_daily_loss_violation = all(pnl >= -self.initial_balance * self.max_daily_loss for pnl in self.daily_pnl.values())
        no_total_loss_violation = total_return >= -self.max_total_loss
        min_trading_days_met = self.trading_days >= self.min_trading_days
        
        ftmo_compliant = (profit_target_reached and no_daily_loss_violation and 
                         no_total_loss_violation and min_trading_days_met)
        
        # Compliance details
        compliance_details = {
            'profit_target_reached': profit_target_reached,
            'no_daily_loss_violation': no_daily_loss_violation,
            'no_total_loss_violation': no_total_loss_violation,
            'min_trading_days_met': min_trading_days_met
        }
        
        return {
            'strategy': strategy_name,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'final_balance': self.current_balance,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'trading_days': self.trading_days,
            'ftmo_compliant': ftmo_compliant,
            'compliance_details': compliance_details,
            'daily_pnl_violations': [date for date, pnl in self.daily_pnl.items() 
                                   if pnl < -self.initial_balance * self.max_daily_loss]
        }

def main():
    """Run comprehensive FTMO backtesting"""
    print("FTMO Challenge Strategy Backtesting")
    print("=" * 50)
    
    # Initialize backtester
    backtester = FTMOBacktester(initial_balance=100000)
    
    # DAX symbols for testing
    symbols = ['^GDAXI', 'SAP.DE', 'SIE.DE', 'ALV.DE', 'BMW.DE', 'BAS.DE', 'ADS.DE', 
               'VOW3.DE', 'DBK.DE', 'IFX.DE', 'MRK.DE']
    
    # Test period (last 2 years for comprehensive testing)
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=730)  # 2 years
    
    print(f"Testing period: {start_date} to {end_date}")
    print(f"Symbols: {symbols}")
    
    # Fetch market data
    print("\nFetching market data...")
    data = backtester.get_market_data(symbols, start_date, end_date)
    
    if data.empty:
        print("No market data available. Exiting.")
        return
    
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    
    # Test multiple strategies
    strategies = [
        (backtester.strategy_mean_reversion_conservative, "Mean Reversion Conservative"),
        (backtester.strategy_momentum_breakout, "Momentum Breakout"),
        (backtester.strategy_trend_following, "Trend Following")
    ]
    
    results = []
    
    for strategy_func, strategy_name in strategies:
        try:
            result = backtester.backtest_strategy(data, strategy_func, strategy_name)
            results.append(result)
        except Exception as e:
            print(f"Error testing {strategy_name}: {e}")
            continue
    
    # Display results
    print("\n" + "=" * 80)
    print("FTMO CHALLENGE BACKTEST RESULTS")
    print("=" * 80)
    
    for result in results:
        print(f"\nStrategy: {result['strategy']}")
        print("-" * 40)
        print(f"Total Return: {result['total_return_pct']:.2f}%")
        print(f"Final Balance: €{result['final_balance']:,.2f}")
        print(f"Total Trades: {result['total_trades']}")
        print(f"Win Rate: {result['win_rate']:.1%}")
        print(f"Profit Factor: {result['profit_factor']:.2f}")
        print(f"Max Drawdown: {result['max_drawdown']:.1%}")
        print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
        print(f"Trading Days: {result['trading_days']}")
        
        print(f"\nFTMO COMPLIANCE: {'✅ PASS' if result['ftmo_compliant'] else '❌ FAIL'}")
        
        compliance = result['compliance_details']
        print(f"  Profit Target (10%): {'✅' if compliance['profit_target_reached'] else '❌'}")
        print(f"  No Daily Loss Violations: {'✅' if compliance['no_daily_loss_violation'] else '❌'}")
        print(f"  No Total Loss Violations: {'✅' if compliance['no_total_loss_violation'] else '❌'}")
        print(f"  Min Trading Days (4): {'✅' if compliance['min_trading_days_met'] else '❌'}")
        
        if result['daily_pnl_violations']:
            print(f"  Daily Loss Violations: {len(result['daily_pnl_violations'])} days")
    
    # Find best FTMO-compliant strategy
    ftmo_compliant_strategies = [r for r in results if r['ftmo_compliant']]
    
    if ftmo_compliant_strategies:
        best_strategy = max(ftmo_compliant_strategies, key=lambda x: x['total_return'])
        print(f"\n🏆 RECOMMENDED STRATEGY: {best_strategy['strategy']}")
        print(f"Expected Return: {best_strategy['total_return_pct']:.2f}%")
        print(f"Risk-Adjusted Performance: {best_strategy['sharpe_ratio']:.2f}")
    else:
        print("\n⚠️  NO STRATEGIES MEET FTMO REQUIREMENTS")
        print("Consider adjusting risk parameters or strategy logic")
        
        # Show best performing strategy even if not compliant
        if results:
            best_non_compliant = max(results, key=lambda x: x['total_return'])
            print(f"\nBest Non-Compliant Strategy: {best_non_compliant['strategy']}")
            print(f"Return: {best_non_compliant['total_return_pct']:.2f}%")

if __name__ == "__main__":
    main()