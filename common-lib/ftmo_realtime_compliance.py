import os
#!/usr/bin/env python3
"""
FTMO Real-Time Compliance Monitor & Risk Management System
Professional implementation for live trading with FTMO rules
"""

import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime, timedelta, time
import json
import warnings
warnings.filterwarnings('ignore')

class FTMORealTimeCompliance:
    def __init__(self, initial_balance=100000, account_id="FTMO_DEMO"):
        self.initial_balance = initial_balance
        self.account_id = account_id

        # FTMO Challenge Rules
        self.profit_target = 0.10  # 10%
        self.max_daily_loss = 0.05  # 5% of initial balance
        self.max_total_loss = 0.10  # 10% of initial balance
        self.min_trading_days = 4

        # Current State
        self.current_balance = initial_balance
        self.current_equity = initial_balance
        self.daily_starting_balance = initial_balance
        self.daily_pnl = 0
        self.total_pnl = 0
        self.trading_days = 0
        self.current_date = datetime.now().date()

        # Position Tracking
        self.positions = {}
        self.daily_trades = 0
        self.max_daily_trades = 5
        self.max_concurrent_positions = 2

        # Risk Limits
        self.max_risk_per_trade = 0.01  # 1%
        self.emergency_stop_triggered = False
        self.compliance_violations = []

    def connect_to_db(self):
        """Connect to trading database"""
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
            print("Warning: Database connection failed. Using manual mode.")
            return None

    def log_compliance_event(self, event_type, message, severity="INFO"):
        """Log compliance events to database"""
        timestamp = datetime.now()
        event = {
            'timestamp': timestamp.isoformat(),
            'account_id': self.account_id,
            'event_type': event_type,
            'message': message,
            'severity': severity,
            'current_balance': self.current_balance,
            'current_equity': self.current_equity,
            'daily_pnl': self.daily_pnl,
            'total_pnl': self.total_pnl
        }

        conn = self.connect_to_db()
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO ftmo_compliance_log
                    (timestamp, account_id, event_type, message, severity,
                     current_balance, current_equity, daily_pnl, total_pnl)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (timestamp, self.account_id, event_type, message, severity,
                      self.current_balance, self.current_equity, self.daily_pnl, self.total_pnl))
                conn.commit()
                conn.close()
            except:
                conn.close()

        print(f"[{severity}] {timestamp.strftime('%H:%M:%S')} - {message}")
        return event

    def check_new_trading_day(self):
        """Check if we've entered a new trading day"""
        today = datetime.now().date()
        if today != self.current_date:
            # Store previous day's results
            if self.daily_pnl != 0:
                self.trading_days += 1
                self.log_compliance_event(
                    "DAY_END",
                    f"Day {self.trading_days} completed. Daily P&L: ${self.daily_pnl:.2f} ({self.daily_pnl/self.initial_balance:.2%})"
                )

            # Reset for new day
            self.current_date = today
            self.daily_starting_balance = self.current_balance
            self.daily_pnl = 0
            self.daily_trades = 0

            self.log_compliance_event("DAY_START", f"New trading day started. Starting balance: ${self.current_balance:.2f}")
            return True
        return False

    def update_equity(self, current_prices):
        """Update real-time equity with floating P&L"""
        floating_pnl = 0

        for symbol, position in self.positions.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]

                if position['type'] == 'long':
                    pnl = (current_price - position['entry_price']) * position['size'] * 100000
                else:
                    pnl = (position['entry_price'] - current_price) * position['size'] * 100000

                floating_pnl += pnl

        self.current_equity = self.current_balance + floating_pnl
        self.daily_pnl = self.current_equity - self.daily_starting_balance
        self.total_pnl = self.current_equity - self.initial_balance

    def check_ftmo_compliance(self):
        """Real-time FTMO compliance check"""
        violations = []

        # Check daily loss limit
        daily_loss_limit = self.initial_balance * self.max_daily_loss
        if self.daily_pnl < -daily_loss_limit:
            violations.append({
                'type': 'DAILY_LOSS_VIOLATION',
                'message': f"Daily loss limit exceeded: ${self.daily_pnl:.2f} (limit: ${-daily_loss_limit:.2f})",
                'severity': 'CRITICAL'
            })
            self.emergency_stop_triggered = True

        # Check total loss limit
        total_loss_limit = self.initial_balance * self.max_total_loss
        if self.total_pnl < -total_loss_limit:
            violations.append({
                'type': 'TOTAL_LOSS_VIOLATION',
                'message': f"Total loss limit exceeded: ${self.total_pnl:.2f} (limit: ${-total_loss_limit:.2f})",
                'severity': 'CRITICAL'
            })
            self.emergency_stop_triggered = True

        # Warning at 80% of limits
        if self.daily_pnl < -daily_loss_limit * 0.8:
            violations.append({
                'type': 'DAILY_LOSS_WARNING',
                'message': f"Daily loss approaching limit: ${self.daily_pnl:.2f} (80% of limit reached)",
                'severity': 'WARNING'
            })

        if self.total_pnl < -total_loss_limit * 0.8:
            violations.append({
                'type': 'TOTAL_LOSS_WARNING',
                'message': f"Total loss approaching limit: ${self.total_pnl:.2f} (80% of limit reached)",
                'severity': 'WARNING'
            })

        # Log violations
        for violation in violations:
            self.log_compliance_event(violation['type'], violation['message'], violation['severity'])
            self.compliance_violations.append({
                'timestamp': datetime.now(),
                **violation
            })

        return len([v for v in violations if v['severity'] == 'CRITICAL']) == 0

    def can_open_position(self, symbol, position_type, entry_price, stop_loss_price):
        """Check if new position can be opened"""
        if self.emergency_stop_triggered:
            self.log_compliance_event("POSITION_REJECTED", "Emergency stop active - no new positions allowed", "WARNING")
            return False, "Emergency stop active"

        if len(self.positions) >= self.max_concurrent_positions:
            return False, f"Maximum concurrent positions reached ({self.max_concurrent_positions})"

        if self.daily_trades >= self.max_daily_trades:
            return False, f"Daily trade limit reached ({self.max_daily_trades})"

        # Calculate position size
        position_size = self.calculate_position_size(entry_price, stop_loss_price)
        if position_size <= 0:
            return False, "Invalid position size"

        return True, "Position approved"

    def calculate_position_size(self, entry_price, stop_loss_price):
        """Calculate FTMO-compliant position size"""
        if stop_loss_price <= 0 or entry_price <= 0:
            return 0

        risk_amount = self.current_equity * self.max_risk_per_trade
        price_risk_pips = abs(entry_price - stop_loss_price) * 10000

        if price_risk_pips == 0:
            return 0

        # $10 per pip per standard lot
        max_lots = risk_amount / (price_risk_pips * 10)

        # Conservative maximum: 0.1 lots per trade
        return min(max_lots, 0.1)

    def open_position(self, symbol, position_type, entry_price, stop_loss_price, take_profit_price=None):
        """Open a new position with FTMO compliance"""
        can_open, reason = self.can_open_position(symbol, position_type, entry_price, stop_loss_price)

        if not can_open:
            self.log_compliance_event("POSITION_REJECTED", f"{symbol} {position_type} rejected: {reason}", "WARNING")
            return False

        position_size = self.calculate_position_size(entry_price, stop_loss_price)

        position = {
            'symbol': symbol,
            'type': position_type,
            'size': position_size,
            'entry_price': entry_price,
            'stop_loss': stop_loss_price,
            'take_profit': take_profit_price,
            'entry_time': datetime.now(),
            'entry_value': position_size * 100000 * entry_price
        }

        self.positions[symbol] = position
        self.daily_trades += 1

        self.log_compliance_event(
            "POSITION_OPENED",
            f"Opened {position_type} {symbol}: {position_size:.3f} lots @ {entry_price:.5f} (Stop: {stop_loss_price:.5f})"
        )

        return True

    def close_position(self, symbol, exit_price, exit_reason="manual"):
        """Close a position and update balances"""
        if symbol not in self.positions:
            return False

        position = self.positions[symbol]

        # Calculate P&L
        if position['type'] == 'long':
            pnl_pips = (exit_price - position['entry_price']) * 10000
        else:
            pnl_pips = (position['entry_price'] - exit_price) * 10000

        pnl_usd = pnl_pips * position['size'] * 10  # $10 per pip per lot

        # Update balances
        self.current_balance += pnl_usd
        self.current_equity = self.current_balance

        duration = datetime.now() - position['entry_time']

        self.log_compliance_event(
            "POSITION_CLOSED",
            f"Closed {position['type']} {symbol}: {pnl_pips:+.1f} pips (${pnl_usd:+.2f}) - {exit_reason.upper()} - Duration: {duration}"
        )

        del self.positions[symbol]
        return True

    def get_compliance_status(self):
        """Get current FTMO compliance status"""
        profit_target_reached = self.total_pnl >= self.initial_balance * self.profit_target
        daily_loss_ok = self.daily_pnl >= -self.initial_balance * self.max_daily_loss
        total_loss_ok = self.total_pnl >= -self.initial_balance * self.max_total_loss
        min_days_met = self.trading_days >= self.min_trading_days

        return {
            'account_id': self.account_id,
            'timestamp': datetime.now().isoformat(),
            'current_balance': self.current_balance,
            'current_equity': self.current_equity,
            'daily_pnl': self.daily_pnl,
            'daily_pnl_pct': self.daily_pnl / self.initial_balance,
            'total_pnl': self.total_pnl,
            'total_pnl_pct': self.total_pnl / self.initial_balance,
            'trading_days': self.trading_days,
            'daily_trades': self.daily_trades,
            'open_positions': len(self.positions),
            'emergency_stop': self.emergency_stop_triggered,
            'compliance': {
                'profit_target_reached': profit_target_reached,
                'daily_loss_ok': daily_loss_ok,
                'total_loss_ok': total_loss_ok,
                'min_trading_days_met': min_days_met,
                'overall_compliant': all([daily_loss_ok, total_loss_ok, not self.emergency_stop_triggered])
            },
            'limits': {
                'max_daily_loss_usd': self.initial_balance * self.max_daily_loss,
                'max_total_loss_usd': self.initial_balance * self.max_total_loss,
                'profit_target_usd': self.initial_balance * self.profit_target,
                'daily_loss_remaining': (self.initial_balance * self.max_daily_loss) + self.daily_pnl,
                'total_loss_remaining': (self.initial_balance * self.max_total_loss) + self.total_pnl
            }
        }

    def generate_compliance_report(self):
        """Generate comprehensive compliance report"""
        status = self.get_compliance_status()

        print("=" * 80)
        print("🏦 FTMO CHALLENGE COMPLIANCE REPORT")
        print("=" * 80)
        print(f"Account ID: {self.account_id}")
        print(f"Report Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        print("💰 ACCOUNT SUMMARY")
        print("-" * 40)
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Current Balance: ${self.current_balance:,.2f}")
        print(f"Current Equity: ${self.current_equity:,.2f}")
        print(f"Total P&L: ${self.total_pnl:+,.2f} ({self.total_pnl/self.initial_balance:+.2%})")
        print(f"Daily P&L: ${self.daily_pnl:+,.2f} ({self.daily_pnl/self.initial_balance:+.2%})")
        print()

        print("📊 TRADING ACTIVITY")
        print("-" * 40)
        print(f"Trading Days: {self.trading_days}")
        print(f"Today's Trades: {self.daily_trades}")
        print(f"Open Positions: {len(self.positions)}")
        print()

        print("🎯 FTMO COMPLIANCE STATUS")
        print("-" * 40)
        compliance = status['compliance']

        profit_status = "✅ REACHED" if compliance['profit_target_reached'] else "❌ NOT REACHED"
        print(f"Profit Target (10%): {profit_status}")
        print(f"  Target: ${status['limits']['profit_target_usd']:,.2f}")
        print(f"  Current: ${self.total_pnl:+,.2f}")
        print()

        daily_status = "✅ COMPLIANT" if compliance['daily_loss_ok'] else "❌ VIOLATED"
        print(f"Daily Loss Limit (5%): {daily_status}")
        print(f"  Limit: ${-status['limits']['max_daily_loss_usd']:,.2f}")
        print(f"  Current: ${self.daily_pnl:+,.2f}")
        print(f"  Remaining: ${status['limits']['daily_loss_remaining']:,.2f}")
        print()

        total_status = "✅ COMPLIANT" if compliance['total_loss_ok'] else "❌ VIOLATED"
        print(f"Total Loss Limit (10%): {total_status}")
        print(f"  Limit: ${-status['limits']['max_total_loss_usd']:,.2f}")
        print(f"  Current: ${self.total_pnl:+,.2f}")
        print(f"  Remaining: ${status['limits']['total_loss_remaining']:,.2f}")
        print()

        days_status = "✅ MET" if compliance['min_trading_days_met'] else "❌ NOT MET"
        print(f"Minimum Trading Days (4): {days_status}")
        print(f"  Required: 4 days")
        print(f"  Completed: {self.trading_days} days")
        print()

        print("🚨 RISK STATUS")
        print("-" * 40)
        if self.emergency_stop_triggered:
            print("❌ EMERGENCY STOP ACTIVE - NO NEW TRADES ALLOWED")
        else:
            print("✅ Normal trading operations")

        print(f"Risk per trade: {self.max_risk_per_trade:.1%}")
        print(f"Max concurrent positions: {self.max_concurrent_positions}")
        print(f"Daily trade limit: {self.max_daily_trades}")
        print()

        if self.positions:
            print("📈 OPEN POSITIONS")
            print("-" * 40)
            for symbol, pos in self.positions.items():
                duration = datetime.now() - pos['entry_time']
                print(f"{symbol}: {pos['type'].upper()} {pos['size']:.3f} lots @ {pos['entry_price']:.5f}")
                print(f"  Stop: {pos['stop_loss']:.5f} | Duration: {duration}")
            print()

        if self.compliance_violations:
            print("⚠️  RECENT VIOLATIONS")
            print("-" * 40)
            recent_violations = sorted(self.compliance_violations, key=lambda x: x['timestamp'], reverse=True)[:5]
            for violation in recent_violations:
                print(f"{violation['timestamp'].strftime('%H:%M:%S')} - {violation['type']}: {violation['message']}")
            print()

        overall_status = "✅ PASSING" if compliance['overall_compliant'] else "❌ FAILING"
        print(f"🏆 OVERALL CHALLENGE STATUS: {overall_status}")
        print("=" * 80)

        return status

def setup_ftmo_compliance_monitoring():
    """Setup FTMO compliance monitoring system"""

    # Create compliance table if it doesn't exist
    conn_string = """
    CREATE TABLE IF NOT EXISTS ftmo_compliance_log (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
        account_id VARCHAR(50) NOT NULL,
        event_type VARCHAR(50) NOT NULL,
        message TEXT NOT NULL,
        severity VARCHAR(20) NOT NULL,
        current_balance DECIMAL(15,2),
        current_equity DECIMAL(15,2),
        daily_pnl DECIMAL(15,2),
        total_pnl DECIMAL(15,2)
    );

    CREATE INDEX IF NOT EXISTS idx_ftmo_compliance_timestamp ON ftmo_compliance_log(timestamp);
    CREATE INDEX IF NOT EXISTS idx_ftmo_compliance_account ON ftmo_compliance_log(account_id);
    """

    try:
        conn = psycopg2.connect(
            host="postgres-service.dax-trading.svc.cluster.local",
            database="finrl_dax",
            user="finrl_user",
            password=os.getenv("POSTGRES_PASSWORD"),
            port="5432"
        )
        cursor = conn.cursor()
        cursor.execute(conn_string)
        conn.commit()
        conn.close()
        print("✅ FTMO compliance monitoring database setup complete")
    except Exception as e:
        print(f"⚠️  Database setup failed: {e}")
        print("📝 Manual mode available without database logging")

def main():
    """Demo of FTMO real-time compliance system"""
    print("🏦 FTMO REAL-TIME COMPLIANCE SYSTEM")
    print("=" * 50)

    # Setup database
    setup_ftmo_compliance_monitoring()

    # Initialize compliance monitor
    ftmo = FTMORealTimeCompliance(initial_balance=100000, account_id="DEMO_ACCOUNT_001")

    print("\n📋 SYSTEM INITIALIZED")
    print(f"Initial Balance: ${ftmo.initial_balance:,}")
    print(f"Max Daily Loss: ${ftmo.initial_balance * ftmo.max_daily_loss:,.2f} ({ftmo.max_daily_loss:.0%})")
    print(f"Max Total Loss: ${ftmo.initial_balance * ftmo.max_total_loss:,.2f} ({ftmo.max_total_loss:.0%})")
    print(f"Profit Target: ${ftmo.initial_balance * ftmo.profit_target:,.2f} ({ftmo.profit_target:.0%})")

    # Demo trading scenarios
    print("\n🎯 DEMO TRADING SCENARIOS")
    print("-" * 30)

    # Scenario 1: Normal trading
    print("\n📈 Scenario 1: Opening EURUSD position")
    success = ftmo.open_position("EURUSD", "long", 1.1000, 1.0950, 1.1100)
    if success:
        # Simulate price movement
        ftmo.update_equity({"EURUSD": 1.1025})  # +25 pips
        ftmo.check_ftmo_compliance()

        # Close position
        ftmo.close_position("EURUSD", 1.1025, "take_profit")

    # Scenario 2: Warning level reached
    print("\n⚠️  Scenario 2: Approaching daily loss limit")
    # Simulate multiple losing trades
    for i in range(3):
        ftmo.open_position(f"TRADE_{i}", "long", 1.2000, 1.1950)
        ftmo.update_equity({f"TRADE_{i}": 1.1945})  # Loss
        ftmo.close_position(f"TRADE_{i}", 1.1945, "stop_loss")

    # Generate final report
    print("\n📊 FINAL COMPLIANCE REPORT")
    ftmo.generate_compliance_report()

if __name__ == "__main__":
    main()