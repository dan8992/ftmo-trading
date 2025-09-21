#!/usr/bin/env python3
"""
Simplified Blind Forward Test - EURUSD Out-of-Sample
"""
import os
import psycopg2
import requests
import json
from datetime import datetime, timedelta
import random

class SimpleForwardTest:
    def __init__(self):
        self.db_config = {
            'host': os.getenv('POSTGRES_HOST', 'postgres-service'),
            'port': 5432,
            'database': os.getenv('POSTGRES_DB', 'dax_trading'),
            'user': os.getenv('POSTGRES_USER', 'finrl_user'),
            'password': os.getenv('POSTGRES_PASSWORD')
        }
        self.account_balance = 100000.0
        self.trades = []
        self.signals_generated = 0

    def connect_db(self):
        return psycopg2.connect(**self.db_config)

    def get_forward_test_data(self):
        """Get EURUSD forward test data"""
        conn = self.connect_db()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT timestamp, symbol, open, high, low, close, volume
                    FROM forward_test_data_1m
                    WHERE symbol = 'EURUSD'
                    ORDER BY timestamp ASC
                """)
                return cur.fetchall()
        finally:
            conn.close()

    def generate_signal(self, symbol, price_data, timestamp):
        """Generate trading signal based on simple momentum"""
        price_change_pct = (price_data['close'] - price_data['open']) / price_data['open']
        volatility = (price_data['high'] - price_data['low']) / price_data['open']

        # Simple momentum strategy
        if price_change_pct > 0.0005 and volatility < 0.003:
            signal_type = 'BUY'
            confidence = min(0.85, 0.6 + abs(price_change_pct) * 200)
            reasoning = f"Upward momentum {price_change_pct:.4f}% with low volatility"
        elif price_change_pct < -0.0005 and volatility < 0.003:
            signal_type = 'SELL'
            confidence = min(0.85, 0.6 + abs(price_change_pct) * 200)
            reasoning = f"Downward momentum {price_change_pct:.4f}% with low volatility"
        else:
            signal_type = 'HOLD'
            confidence = 0.4
            reasoning = f"Sideways movement"

        return {
            'signal_type': signal_type,
            'confidence': confidence,
            'reasoning': reasoning,
            'timestamp': timestamp,
            'symbol': symbol
        }

    def store_signal(self, signal):
        """Store signal in database"""
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
                self.signals_generated += 1
        finally:
            conn.close()

    def execute_trade(self, signal, current_price):
        """Execute paper trade"""
        if signal['signal_type'] == 'HOLD' or signal['confidence'] < 0.65:
            return None

        # Simple position sizing
        position_size = 1000  # $1000 per trade

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
        return trade

    def close_random_trades(self, current_price, timestamp):
        """Randomly close some trades"""
        open_trades = [t for t in self.trades if t['status'] == 'OPEN']

        for trade in open_trades:
            if random.random() < 0.03:  # 3% chance to close
                self.close_trade(trade, current_price, timestamp)

    def close_trade(self, trade, exit_price, timestamp):
        """Close trade and calculate P&L"""
        if trade['signal_type'] == 'BUY':
            pnl = (exit_price - trade['entry_price']) * trade['position_size'] * 10000  # Convert to dollars
        else:
            pnl = (trade['entry_price'] - exit_price) * trade['position_size'] * 10000

        trade['exit_price'] = exit_price
        trade['pnl'] = pnl
        trade['status'] = 'CLOSED'
        self.account_balance += pnl

        print(f"INFO:__main__:Closed trade {trade['id']}: P&L = ${pnl:.2f}, Balance = ${self.account_balance:,.2f}")

    def run_forward_test(self):
        """Execute forward test"""
        print("🚀 Starting EURUSD Blind Forward Test")
        print("=" * 60)

        market_data = self.get_forward_test_data()
        print(f"📊 Processing {len(market_data):,} EURUSD data points")
        print(f"💰 Starting balance: ${self.account_balance:,.2f}")
        print("-" * 60)

        processed = 0
        for timestamp, symbol, open_price, high, low, close, volume in market_data:
            price_data = {
                'open': float(open_price),
                'high': float(high),
                'low': float(low),
                'close': float(close),
                'volume': int(volume)
            }

            # Generate and store signal
            signal = self.generate_signal(symbol, price_data, timestamp)
            self.store_signal(signal)

            # Execute trade
            trade = self.execute_trade(signal, price_data['close'])
            if trade:
                print(f"INFO:__main__:Executed trade {trade['id']}: {trade['signal_type']} @ {trade['entry_price']:.5f}")

            # Close some trades
            self.close_random_trades(price_data['close'], timestamp)

            processed += 1
            if processed % 5000 == 0:
                print(f"📈 Progress: {processed:,}/{len(market_data):,} ({100*processed/len(market_data):.1f}%)")

        self.generate_report()

    def generate_report(self):
        """Generate test results"""
        total_trades = len(self.trades)
        closed_trades = [t for t in self.trades if t['status'] == 'CLOSED']
        total_pnl = sum(t.get('pnl', 0) for t in closed_trades)
        win_rate = len([t for t in closed_trades if t.get('pnl', 0) > 0]) / len(closed_trades) if closed_trades else 0

        print(f"\n📋 FORWARD TEST RESULTS")
        print("=" * 60)
        print(f"📊 Signals Generated: {self.signals_generated:,}")
        print(f"💼 Total Trades: {total_trades}")
        print(f"✅ Closed Trades: {len(closed_trades)}")
        print(f"📈 Total P&L: ${total_pnl:,.2f}")
        print(f"🎯 Win Rate: {win_rate:.1%}")
        print(f"💰 Final Balance: ${self.account_balance:,.2f}")
        print(f"📊 Return: {((self.account_balance - 100000) / 100000 * 100):+.2f}%")

if __name__ == "__main__":
    test = SimpleForwardTest()
    test.run_forward_test()