#!/usr/bin/env python3
"""
Blind Forward Test - Out-of-Sample Backtester
Replays the last 15 days of market data minute-by-minute to test system robustness
"""
import os
import sys
import time
import psycopg2
import requests
import json
from datetime import datetime, timedelta
import random

class ForwardTestBacktester:
    def __init__(self):
        self.db_config = {
            'host': 'postgres-service',
            'port': 5432,
            'database': 'dax_trading',
            'user': 'finrl_user',
            'password': os.getenv('POSTGRES_PASSWORD')
        }
        self.llm_service_url = "http://dax-llm-service:8000"
        self.account_balance = 100000.0
        self.max_daily_loss = 0.05  # 5% FTMO daily loss limit
        self.max_total_loss = 0.10  # 10% FTMO total loss limit
        self.max_risk_per_trade = 0.02  # 2% risk per trade
        self.trades = []
        self.daily_pnl = {}
        
    def connect_db(self):
        """Connect to PostgreSQL database"""
        return psycopg2.connect(**self.db_config)
    
    def get_market_data_chronologically(self):
        """Get forward test data in chronological order"""
        conn = self.connect_db()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT timestamp, symbol, open, high, low, close, volume
                    FROM forward_test_data_1m 
                    ORDER BY timestamp ASC, symbol ASC
                """)
                return cur.fetchall()
        finally:
            conn.close()
    
    def generate_signal_with_llm(self, symbol, price_data, timestamp):
        """Generate trading signal using LLM service"""
        try:
            # Prepare market context
            market_context = {
                "symbol": symbol,
                "timestamp": timestamp.isoformat(),
                "current_price": price_data['close'],
                "volume": price_data['volume'],
                "price_change": price_data['close'] - price_data['open'],
                "high_low_range": price_data['high'] - price_data['low']
            }
            
            # Call LLM service
            response = requests.post(
                f"{self.llm_service_url}/generate_signal",
                json=market_context,
                timeout=30
            )
            
            if response.status_code == 200:
                signal_data = response.json()
                return {
                    'signal_type': signal_data.get('signal', 'HOLD'),
                    'confidence': signal_data.get('confidence', 0.5),
                    'reasoning': signal_data.get('reasoning', 'LLM analysis'),
                    'timestamp': timestamp,
                    'symbol': symbol
                }
            else:
                print(f"⚠️ LLM service error: {response.status_code}")
                return self.generate_fallback_signal(symbol, price_data, timestamp)
                
        except Exception as e:
            print(f"⚠️ LLM service unavailable: {e}")
            return self.generate_fallback_signal(symbol, price_data, timestamp)
    
    def generate_fallback_signal(self, symbol, price_data, timestamp):
        """Generate fallback signal when LLM is unavailable"""
        # Simple technical analysis based fallback
        price_change_pct = (price_data['close'] - price_data['open']) / price_data['open']
        volatility = (price_data['high'] - price_data['low']) / price_data['open']
        
        # Generate signal based on momentum and volatility
        if price_change_pct > 0.001 and volatility < 0.005:  # Rising with low volatility
            signal_type = 'BUY'
            confidence = min(0.8, 0.5 + abs(price_change_pct) * 100)
            reasoning = f"Upward momentum {price_change_pct:.4f}% with low volatility {volatility:.4f}%"
        elif price_change_pct < -0.001 and volatility < 0.005:  # Falling with low volatility
            signal_type = 'SELL'
            confidence = min(0.8, 0.5 + abs(price_change_pct) * 100)
            reasoning = f"Downward momentum {price_change_pct:.4f}% with low volatility {volatility:.4f}%"
        else:
            signal_type = 'HOLD'
            confidence = 0.3
            reasoning = f"Unclear direction, high volatility {volatility:.4f}%"
        
        return {
            'signal_type': signal_type,
            'confidence': confidence,
            'reasoning': reasoning,
            'timestamp': timestamp,
            'symbol': symbol
        }
    
    def store_signal(self, signal):
        """Store generated signal in database"""
        conn = self.connect_db()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO trading_signals (timestamp, symbol, signal_type, confidence, reasoning)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    signal['timestamp'],
                    signal['symbol'],
                    signal['signal_type'],
                    signal['confidence'],
                    signal['reasoning']
                ))
                conn.commit()
        finally:
            conn.close()
    
    def execute_trade(self, signal, current_price):
        """Execute paper trade based on signal"""
        if signal['signal_type'] == 'HOLD' or signal['confidence'] < 0.6:
            return None
        
        # Calculate position size based on 2% risk per trade
        risk_amount = self.account_balance * self.max_risk_per_trade
        
        # Assume 50 pip stop loss for forex
        stop_loss_pips = 50
        pip_value = 0.0001 if signal['symbol'] == 'EURUSD' else 0.0001
        stop_loss_amount = stop_loss_pips * pip_value * current_price
        
        # Position size calculation
        position_size = min(risk_amount / stop_loss_amount, self.account_balance * 0.1)
        
        trade = {
            'id': len(self.trades) + 1,
            'symbol': signal['symbol'],
            'signal_type': signal['signal_type'],
            'entry_price': current_price,
            'position_size': position_size,
            'timestamp': signal['timestamp'],
            'confidence': signal['confidence'],
            'status': 'OPEN'
        }
        
        self.trades.append(trade)
        print(f"INFO:__main__:Executed trade {trade['id']}: {trade['signal_type']} {trade['symbol']} @ {current_price:.5f} (size: ${position_size:,.0f})")
        return trade
    
    def close_random_trades(self, current_data):
        """Randomly close some open trades to simulate trading activity"""
        open_trades = [t for t in self.trades if t['status'] == 'OPEN']
        
        for trade in open_trades:
            # 5% chance to close any open trade each minute
            if random.random() < 0.05:
                self.close_trade(trade, current_data[trade['symbol']]['close'])
    
    def close_trade(self, trade, exit_price):
        """Close an open trade and calculate P&L"""
        if trade['signal_type'] == 'BUY':
            pnl = (exit_price - trade['entry_price']) * trade['position_size']
        else:  # SELL
            pnl = (trade['entry_price'] - exit_price) * trade['position_size']
        
        trade['exit_price'] = exit_price
        trade['pnl'] = pnl
        trade['status'] = 'CLOSED'
        
        self.account_balance += pnl
        
        # Track daily P&L
        trade_date = trade['timestamp'].date()
        if trade_date not in self.daily_pnl:
            self.daily_pnl[trade_date] = 0
        self.daily_pnl[trade_date] += pnl
        
        print(f"INFO:__main__:Closed trade {trade['id']}: P&L = ${pnl:.2f}, Balance = ${self.account_balance:,.2f}")
    
    def check_ftmo_compliance(self, current_date):
        """Check FTMO compliance rules"""
        # Daily loss check
        daily_loss = self.daily_pnl.get(current_date, 0)
        daily_loss_pct = abs(daily_loss) / 100000.0 if daily_loss < 0 else 0
        
        # Total loss check
        total_loss = 100000.0 - self.account_balance
        total_loss_pct = total_loss / 100000.0 if total_loss > 0 else 0
        
        is_compliant = (daily_loss_pct <= self.max_daily_loss and 
                       total_loss_pct <= self.max_total_loss)
        
        # Store compliance log
        self.store_compliance_log(current_date, is_compliant, daily_loss_pct, total_loss_pct)
        
        return is_compliant
    
    def store_compliance_log(self, date, is_compliant, daily_loss_pct, total_loss_pct):
        """Store FTMO compliance check results"""
        conn = self.connect_db()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO ftmo_compliance_log 
                    (timestamp, account_balance, daily_loss_pct, total_loss_pct, is_compliant)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (timestamp) DO UPDATE SET
                        account_balance = EXCLUDED.account_balance,
                        daily_loss_pct = EXCLUDED.daily_loss_pct,
                        total_loss_pct = EXCLUDED.total_loss_pct,
                        is_compliant = EXCLUDED.is_compliant
                """, (date, self.account_balance, daily_loss_pct, total_loss_pct, is_compliant))
                conn.commit()
        finally:
            conn.close()
    
    def run_forward_test(self):
        """Execute the blind forward test"""
        print("🚀 Starting Blind Forward Test (Out-of-Sample)")
        print("=" * 60)
        
        # Get all forward test data chronologically
        market_data = self.get_market_data_chronologically()
        
        print(f"📊 Processing {len(market_data)} market data points...")
        print(f"💰 Starting balance: ${self.account_balance:,.2f}")
        print("-" * 60)
        
        current_minute_data = {}
        processed_count = 0
        
        for timestamp, symbol, open_price, high, low, close, volume in market_data:
            # Group data by timestamp (minute)
            if timestamp not in current_minute_data:
                # Process previous minute if exists
                if current_minute_data:
                    self.process_minute(list(current_minute_data.keys())[0], 
                                      list(current_minute_data.values())[0])
                
                current_minute_data = {timestamp: {}}
            
            # Add symbol data to current minute
            current_minute_data[timestamp][symbol] = {
                'open': float(open_price),
                'high': float(high),
                'low': float(low),
                'close': float(close),
                'volume': int(volume)
            }
            
            processed_count += 1
            if processed_count % 5000 == 0:
                print(f"📈 Processed {processed_count:,} data points, Balance: ${self.account_balance:,.2f}")
        
        # Process final minute
        if current_minute_data:
            self.process_minute(list(current_minute_data.keys())[0], 
                              list(current_minute_data.values())[0])
        
        print("\n🎯 Forward Test Completed!")
        self.generate_forward_test_report()
    
    def process_minute(self, timestamp, symbol_data):
        """Process one minute of market data"""
        # Generate signals for each symbol
        for symbol, price_data in symbol_data.items():
            # Generate signal
            signal = self.generate_signal_with_llm(symbol, price_data, timestamp)
            
            # Store signal
            self.store_signal(signal)
            
            # Execute trade if signal is strong enough
            trade = self.execute_trade(signal, price_data['close'])
        
        # Close some random trades
        self.close_random_trades(symbol_data)
        
        # Check FTMO compliance daily
        self.check_ftmo_compliance(timestamp.date())
    
    def generate_forward_test_report(self):
        """Generate forward test summary report"""
        total_trades = len(self.trades)
        closed_trades = [t for t in self.trades if t['status'] == 'CLOSED']
        total_pnl = sum(t.get('pnl', 0) for t in closed_trades)
        win_rate = len([t for t in closed_trades if t.get('pnl', 0) > 0]) / len(closed_trades) if closed_trades else 0
        
        print(f"\n📋 FORWARD TEST SUMMARY")
        print("=" * 60)
        print(f"💼 Total Trades: {total_trades}")
        print(f"✅ Closed Trades: {len(closed_trades)}")
        print(f"📈 Total P&L: ${total_pnl:,.2f}")
        print(f"🎯 Win Rate: {win_rate:.1%}")
        print(f"💰 Final Balance: ${self.account_balance:,.2f}")
        print(f"📊 Return: {((self.account_balance - 100000) / 100000 * 100):+.2f}%")

if __name__ == "__main__":
    backtester = ForwardTestBacktester()
    backtester.run_forward_test()