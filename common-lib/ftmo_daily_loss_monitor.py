import os
#!/usr/bin/env python3
"""
FTMO Daily Loss Limit Monitor
Implements real-time monitoring of daily loss limits per FTMO rules
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime, date, timedelta
from collections import defaultdict
import json

class FTMODailyLossMonitor:
    """
    Monitor daily P&L to enforce FTMO 5% daily loss limit
    """

    def __init__(self, account_balance: float, daily_loss_limit: float = 0.05):
        """
        Initialize daily loss monitor

        Args:
            account_balance: Current account balance
            daily_loss_limit: Maximum daily loss as percentage (default 5% for FTMO)
        """
        self.account_balance = account_balance
        self.daily_loss_limit = daily_loss_limit

        # Track daily P&L by date
        self.daily_pnl = defaultdict(float)

        # Track individual trades by date
        self.daily_trades = defaultdict(list)

        # Trading suspension status
        self.trading_suspended = False
        self.suspension_date = None

        # Violation history
        self.violation_history = []

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def update_daily_pnl(self, trade_pnl: float, trade_date: date,
                        trade_id: str = None, symbol: str = None) -> Dict:
        """
        Update daily P&L and check limits

        Args:
            trade_pnl: P&L from completed trade (positive = profit, negative = loss)
            trade_date: Date of the trade
            trade_id: Optional trade identifier
            symbol: Optional symbol for the trade

        Returns:
            Dict with status and limit information
        """
        # Add to daily P&L tracking
        self.daily_pnl[trade_date] += trade_pnl

        # Record trade details
        trade_record = {
            "trade_id": trade_id,
            "symbol": symbol,
            "pnl": trade_pnl,
            "timestamp": datetime.utcnow(),
            "cumulative_daily_pnl": self.daily_pnl[trade_date]
        }
        self.daily_trades[trade_date].append(trade_record)

        # Check if this is a new day and reset suspension if needed
        today = date.today()
        if self.trading_suspended and self.suspension_date != today:
            self.reset_daily_suspension()

        # Calculate current daily loss
        current_daily_pnl = self.daily_pnl[trade_date]
        daily_loss = abs(min(0, current_daily_pnl))  # Only count negative P&L as loss
        daily_loss_pct = daily_loss / self.account_balance

        # Check daily loss limit
        limit_exceeded = daily_loss_pct >= self.daily_loss_limit

        if limit_exceeded and not self.trading_suspended:
            self._trigger_daily_loss_violation(trade_date, daily_loss, daily_loss_pct)

        # Calculate remaining daily capacity
        remaining_capacity = self._calculate_remaining_daily_capacity(trade_date)

        result = {
            "trade_date": trade_date,
            "daily_pnl": current_daily_pnl,
            "daily_loss": daily_loss,
            "daily_loss_percentage": daily_loss_pct * 100,
            "limit_percentage": self.daily_loss_limit * 100,
            "limit_exceeded": limit_exceeded,
            "trading_suspended": self.trading_suspended,
            "remaining_capacity": remaining_capacity,
            "trades_today": len(self.daily_trades[trade_date]),
            "status": self._get_status_message(daily_loss_pct)
        }

        self.logger.info(f"Daily P&L update for {trade_date}: ${current_daily_pnl:.2f} "
                        f"({daily_loss_pct*100:.2f}% loss), Status: {result['status']}")

        return result

    def _trigger_daily_loss_violation(self, violation_date: date, loss_amount: float,
                                    loss_percentage: float):
        """
        Handle daily loss limit violation
        """
        self.trading_suspended = True
        self.suspension_date = violation_date

        violation_record = {
            "date": violation_date,
            "loss_amount": loss_amount,
            "loss_percentage": loss_percentage,
            "account_balance": self.account_balance,
            "timestamp": datetime.utcnow(),
            "trades_count": len(self.daily_trades[violation_date])
        }

        self.violation_history.append(violation_record)

        self.logger.critical(f"DAILY LOSS LIMIT EXCEEDED on {violation_date}: "
                           f"${loss_amount:.2f} ({loss_percentage*100:.2f}%) - TRADING SUSPENDED")

    def reset_daily_suspension(self):
        """
        Reset trading suspension for new trading day
        """
        if self.trading_suspended:
            self.logger.info("Daily trading suspension reset for new trading day")
            self.trading_suspended = False
            self.suspension_date = None

    def _calculate_remaining_daily_capacity(self, trade_date: date) -> float:
        """
        Calculate remaining risk capacity for the day

        Args:
            trade_date: Date to calculate capacity for

        Returns:
            Remaining capacity in USD
        """
        current_daily_pnl = self.daily_pnl[trade_date]
        max_daily_loss = self.account_balance * self.daily_loss_limit

        if current_daily_pnl >= 0:
            # If in profit, full capacity available
            return max_daily_loss
        else:
            # If in loss, calculate remaining capacity
            current_loss = abs(current_daily_pnl)
            return max(0, max_daily_loss - current_loss)

    def _get_status_message(self, daily_loss_pct: float) -> str:
        """
        Get status message based on current daily loss
        """
        loss_ratio = daily_loss_pct / self.daily_loss_limit

        if self.trading_suspended:
            return "TRADING_SUSPENDED"
        elif loss_ratio >= 0.9:
            return "CRITICAL_RISK"
        elif loss_ratio >= 0.7:
            return "HIGH_RISK"
        elif loss_ratio >= 0.5:
            return "MODERATE_RISK"
        elif loss_ratio >= 0.3:
            return "LOW_RISK"
        else:
            return "SAFE"

    def can_take_trade(self, potential_loss: float, trade_date: date = None) -> Dict:
        """
        Check if a new trade can be taken given potential loss

        Args:
            potential_loss: Maximum potential loss from the trade
            trade_date: Date of potential trade (default: today)

        Returns:
            Dict with permission status and details
        """
        if trade_date is None:
            trade_date = date.today()

        # Check if trading is suspended
        if self.trading_suspended and self.suspension_date == trade_date:
            return {
                "can_trade": False,
                "reason": "DAILY_LIMIT_EXCEEDED",
                "remaining_capacity": 0.0
            }

        # Calculate remaining capacity
        remaining_capacity = self._calculate_remaining_daily_capacity(trade_date)

        # Check if potential loss exceeds remaining capacity
        can_trade = potential_loss <= remaining_capacity

        return {
            "can_trade": can_trade,
            "reason": "OK" if can_trade else "INSUFFICIENT_CAPACITY",
            "remaining_capacity": remaining_capacity,
            "potential_loss": potential_loss,
            "capacity_after_trade": remaining_capacity - potential_loss if can_trade else None
        }

    def get_daily_summary(self, target_date: date = None) -> Dict:
        """
        Get comprehensive daily summary

        Args:
            target_date: Date to get summary for (default: today)

        Returns:
            Dict with daily summary information
        """
        if target_date is None:
            target_date = date.today()

        daily_pnl = self.daily_pnl[target_date]
        daily_trades = self.daily_trades[target_date]

        # Calculate trade statistics
        winning_trades = [t for t in daily_trades if t["pnl"] > 0]
        losing_trades = [t for t in daily_trades if t["pnl"] < 0]

        daily_loss = abs(min(0, daily_pnl))
        daily_loss_pct = daily_loss / self.account_balance

        return {
            "date": target_date,
            "daily_pnl": daily_pnl,
            "daily_loss": daily_loss,
            "daily_loss_percentage": daily_loss_pct * 100,
            "total_trades": len(daily_trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(winning_trades) / len(daily_trades) * 100 if daily_trades else 0,
            "largest_win": max([t["pnl"] for t in winning_trades], default=0),
            "largest_loss": min([t["pnl"] for t in losing_trades], default=0),
            "remaining_capacity": self._calculate_remaining_daily_capacity(target_date),
            "trading_suspended": self.trading_suspended and self.suspension_date == target_date,
            "status": self._get_status_message(daily_loss_pct)
        }

    def get_weekly_summary(self, end_date: date = None) -> Dict:
        """
        Get weekly trading summary

        Args:
            end_date: End date for week (default: today)

        Returns:
            Dict with weekly summary
        """
        if end_date is None:
            end_date = date.today()

        start_date = end_date - timedelta(days=6)  # 7 days including end_date

        weekly_pnl = 0.0
        weekly_trades = 0
        trading_days = 0
        violation_days = 0

        for i in range(7):
            check_date = start_date + timedelta(days=i)
            if check_date in self.daily_pnl:
                weekly_pnl += self.daily_pnl[check_date]
                if self.daily_trades[check_date]:  # Has trades
                    trading_days += 1
                    weekly_trades += len(self.daily_trades[check_date])

                # Check for violations
                daily_loss = abs(min(0, self.daily_pnl[check_date]))
                if daily_loss / self.account_balance >= self.daily_loss_limit:
                    violation_days += 1

        return {
            "period": f"{start_date} to {end_date}",
            "weekly_pnl": weekly_pnl,
            "weekly_return_pct": (weekly_pnl / self.account_balance) * 100,
            "trading_days": trading_days,
            "total_trades": weekly_trades,
            "violation_days": violation_days,
            "avg_daily_pnl": weekly_pnl / 7,
            "max_daily_loss": max([abs(min(0, self.daily_pnl[start_date + timedelta(days=i)]))
                                 for i in range(7)], default=0)
        }

    def update_account_balance(self, new_balance: float):
        """
        Update account balance (affects daily loss limits)

        Args:
            new_balance: New account balance
        """
        old_balance = self.account_balance
        self.account_balance = new_balance
        self.logger.info(f"Account balance updated: ${old_balance:.2f} → ${new_balance:.2f}")

    def export_data(self) -> Dict:
        """
        Export all monitoring data for backup/analysis
        """
        return {
            "account_balance": self.account_balance,
            "daily_loss_limit": self.daily_loss_limit,
            "daily_pnl": dict(self.daily_pnl),
            "daily_trades": {str(k): v for k, v in self.daily_trades.items()},
            "violation_history": self.violation_history,
            "trading_suspended": self.trading_suspended,
            "suspension_date": str(self.suspension_date) if self.suspension_date else None,
            "export_timestamp": datetime.utcnow().isoformat()
        }

# Test the daily loss monitor
if __name__ == "__main__":
    # Initialize with $100,000 account
    monitor = FTMODailyLossMonitor(account_balance=100000.0)

    # Simulate some trades
    today = date.today()

    print("FTMO Daily Loss Monitor Test:")
    print("=" * 50)

    # Test trade 1: Small loss
    result1 = monitor.update_daily_pnl(-500, today, "TRADE_001", "EURUSD")
    print(f"Trade 1 - Loss $500: Status = {result1['status']}, Can Trade = {not result1['trading_suspended']}")

    # Test trade 2: Large loss (approaching limit)
    result2 = monitor.update_daily_pnl(-4000, today, "TRADE_002", "GBPUSD")
    print(f"Trade 2 - Loss $4000: Status = {result2['status']}, Daily Loss = {result2['daily_loss_percentage']:.2f}%")

    # Test trade 3: This should trigger violation
    result3 = monitor.update_daily_pnl(-1000, today, "TRADE_003", "USDJPY")
    print(f"Trade 3 - Loss $1000: Status = {result3['status']}, Trading Suspended = {result3['trading_suspended']}")

    # Test if we can take another trade
    trade_check = monitor.can_take_trade(500, today)
    print(f"Can take new trade with $500 risk: {trade_check['can_trade']}")

    # Get daily summary
    summary = monitor.get_daily_summary(today)
    print(f"\nDaily Summary:")
    print(f"  Total P&L: ${summary['daily_pnl']:.2f}")
    print(f"  Daily Loss: {summary['daily_loss_percentage']:.2f}%")
    print(f"  Status: {summary['status']}")
    print(f"  Trades: {summary['total_trades']} (Win Rate: {summary['win_rate']:.1f}%)")