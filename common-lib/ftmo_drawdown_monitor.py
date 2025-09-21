import os
#!/usr/bin/env python3
"""
FTMO Total Drawdown Monitor
Implements real-time monitoring of maximum drawdown per FTMO rules
"""
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date
from collections import namedtuple
import math

# Data structures
DrawdownEvent = namedtuple('DrawdownEvent', ['timestamp', 'balance', 'peak', 'drawdown_pct', 'drawdown_amount'])
EquityPoint = namedtuple('EquityPoint', ['timestamp', 'balance', 'pnl_change'])

class FTMODrawdownMonitor:
    """
    Monitor account drawdown to enforce FTMO 10% maximum drawdown limit
    """

    def __init__(self, initial_balance: float, max_drawdown: float = 0.10):
        """
        Initialize drawdown monitor

        Args:
            initial_balance: Starting account balance
            max_drawdown: Maximum drawdown allowed (default 10% for FTMO)
        """
        self.initial_balance = initial_balance
        self.max_drawdown = max_drawdown

        # Current account state
        self.current_balance = initial_balance
        self.peak_balance = initial_balance
        self.trough_balance = initial_balance

        # Drawdown tracking
        self.current_drawdown = 0.0
        self.max_drawdown_reached = 0.0
        self.in_drawdown_period = False

        # Historical tracking
        self.equity_curve = []
        self.drawdown_events = []
        self.peak_history = []

        # Violation status
        self.drawdown_violated = False
        self.violation_timestamp = None

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Record initial state
        self._record_equity_point(initial_balance, 0.0)

    def update_balance(self, new_balance: float, timestamp: datetime = None) -> Dict:
        """
        Update account balance and check drawdown limits

        Args:
            new_balance: New account balance
            timestamp: Timestamp of balance update (default: now)

        Returns:
            Dict with drawdown status and metrics
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        # Calculate P&L change
        pnl_change = new_balance - self.current_balance

        # Update current balance
        previous_balance = self.current_balance
        self.current_balance = new_balance

        # Update peak balance if new high
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance
            self.peak_history.append({
                'timestamp': timestamp,
                'balance': new_balance,
                'previous_peak': self.peak_balance
            })

            # End drawdown period if we hit new peak
            if self.in_drawdown_period:
                self._end_drawdown_period(timestamp)

        # Update trough balance if new low
        if new_balance < self.trough_balance:
            self.trough_balance = new_balance

        # Calculate current drawdown from peak
        self.current_drawdown = (self.peak_balance - new_balance) / self.initial_balance

        # Update maximum drawdown reached
        if self.current_drawdown > self.max_drawdown_reached:
            self.max_drawdown_reached = self.current_drawdown

        # Check if we're in a drawdown period
        if not self.in_drawdown_period and new_balance < self.peak_balance:
            self._start_drawdown_period(timestamp)

        # Check for violation
        violation_occurred = False
        if self.current_drawdown >= self.max_drawdown and not self.drawdown_violated:
            violation_occurred = True
            self._trigger_drawdown_violation(timestamp)

        # Record equity point
        self._record_equity_point(new_balance, pnl_change, timestamp)

        # Calculate remaining capacity
        remaining_capacity = self._calculate_remaining_capacity()

        result = {
            "timestamp": timestamp,
            "current_balance": new_balance,
            "peak_balance": self.peak_balance,
            "current_drawdown": self.current_drawdown * 100,
            "max_drawdown_limit": self.max_drawdown * 100,
            "max_drawdown_reached": self.max_drawdown_reached * 100,
            "remaining_capacity": remaining_capacity,
            "in_drawdown_period": self.in_drawdown_period,
            "violation_occurred": violation_occurred,
            "drawdown_violated": self.drawdown_violated,
            "pnl_change": pnl_change,
            "status": self._get_status_message()
        }

        self.logger.info(f"Balance update: ${new_balance:.2f} (DD: {self.current_drawdown*100:.2f}%), "
                        f"Status: {result['status']}")

        return result

    def _start_drawdown_period(self, timestamp: datetime):
        """
        Start tracking a new drawdown period
        """
        self.in_drawdown_period = True
        self.logger.info(f"Drawdown period started at {timestamp}")

    def _end_drawdown_period(self, timestamp: datetime):
        """
        End current drawdown period and record the event
        """
        if self.in_drawdown_period:
            # Calculate the drawdown that just ended
            drawdown_pct = self.max_drawdown_reached
            drawdown_amount = self.initial_balance * drawdown_pct

            # Record the drawdown event
            event = DrawdownEvent(
                timestamp=timestamp,
                balance=self.current_balance,
                peak=self.peak_balance,
                drawdown_pct=drawdown_pct,
                drawdown_amount=drawdown_amount
            )
            self.drawdown_events.append(event)

            self.in_drawdown_period = False
            self.logger.info(f"Drawdown period ended: Max DD was {drawdown_pct*100:.2f}%")

    def _trigger_drawdown_violation(self, timestamp: datetime):
        """
        Handle maximum drawdown violation
        """
        self.drawdown_violated = True
        self.violation_timestamp = timestamp

        violation_amount = self.initial_balance * self.current_drawdown

        self.logger.critical(f"MAXIMUM DRAWDOWN VIOLATED at {timestamp}: "
                           f"{self.current_drawdown*100:.2f}% (${violation_amount:.2f}) - "
                           f"ACCOUNT TERMINATED")

    def _calculate_remaining_capacity(self) -> float:
        """
        Calculate remaining drawdown capacity in USD

        Returns:
            Remaining capacity before violation
        """
        max_loss_allowed = self.initial_balance * self.max_drawdown
        current_loss = self.initial_balance - self.current_balance
        return max(0, max_loss_allowed - current_loss)

    def _get_status_message(self) -> str:
        """
        Get status message based on current drawdown
        """
        if self.drawdown_violated:
            return "VIOLATION_OCCURRED"

        drawdown_ratio = self.current_drawdown / self.max_drawdown

        if drawdown_ratio >= 0.95:
            return "CRITICAL_DANGER"
        elif drawdown_ratio >= 0.8:
            return "HIGH_DANGER"
        elif drawdown_ratio >= 0.6:
            return "MODERATE_RISK"
        elif drawdown_ratio >= 0.4:
            return "LOW_RISK"
        elif drawdown_ratio >= 0.2:
            return "MINIMAL_RISK"
        else:
            return "SAFE"

    def _record_equity_point(self, balance: float, pnl_change: float, timestamp: datetime = None):
        """
        Record equity curve point
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        point = EquityPoint(
            timestamp=timestamp,
            balance=balance,
            pnl_change=pnl_change
        )
        self.equity_curve.append(point)

    def can_take_trade(self, potential_loss: float) -> Dict:
        """
        Check if a trade can be taken given potential maximum loss

        Args:
            potential_loss: Maximum potential loss from the trade

        Returns:
            Dict with permission status and details
        """
        # Can't trade if already violated
        if self.drawdown_violated:
            return {
                "can_trade": False,
                "reason": "DRAWDOWN_VIOLATION",
                "remaining_capacity": 0.0
            }

        # Calculate remaining capacity
        remaining_capacity = self._calculate_remaining_capacity()

        # Check if potential loss exceeds remaining capacity
        can_trade = potential_loss <= remaining_capacity

        # Additional safety buffer (recommended)
        safety_buffer = remaining_capacity * 0.1  # Keep 10% buffer
        safe_capacity = remaining_capacity - safety_buffer
        is_safe = potential_loss <= safe_capacity

        return {
            "can_trade": can_trade,
            "is_safe": is_safe,
            "reason": "OK" if can_trade else "INSUFFICIENT_CAPACITY",
            "remaining_capacity": remaining_capacity,
            "safe_capacity": safe_capacity,
            "potential_loss": potential_loss,
            "balance_after_loss": self.current_balance - potential_loss
        }

    def get_drawdown_metrics(self) -> Dict:
        """
        Get comprehensive drawdown metrics

        Returns:
            Dict with detailed drawdown analysis
        """
        # Calculate recovery statistics
        total_return = (self.current_balance - self.initial_balance) / self.initial_balance

        # Calculate time in drawdown
        total_periods = len(self.equity_curve)
        drawdown_periods = sum(1 for point in self.equity_curve
                             if point.balance < self.peak_balance)
        time_in_drawdown_pct = (drawdown_periods / total_periods * 100) if total_periods > 0 else 0

        # Calculate average drawdown
        avg_drawdown = sum(event.drawdown_pct for event in self.drawdown_events) / len(self.drawdown_events) if self.drawdown_events else 0

        return {
            "current_balance": self.current_balance,
            "initial_balance": self.initial_balance,
            "peak_balance": self.peak_balance,
            "total_return_pct": total_return * 100,
            "current_drawdown_pct": self.current_drawdown * 100,
            "max_drawdown_reached_pct": self.max_drawdown_reached * 100,
            "max_drawdown_limit_pct": self.max_drawdown * 100,
            "remaining_capacity": self._calculate_remaining_capacity(),
            "time_in_drawdown_pct": time_in_drawdown_pct,
            "drawdown_events_count": len(self.drawdown_events),
            "avg_drawdown_pct": avg_drawdown * 100,
            "in_drawdown_period": self.in_drawdown_period,
            "drawdown_violated": self.drawdown_violated,
            "status": self._get_status_message()
        }

    def get_equity_curve_data(self, last_n_points: int = None) -> List[Dict]:
        """
        Get equity curve data for plotting/analysis

        Args:
            last_n_points: Number of recent points to return (default: all)

        Returns:
            List of equity curve points
        """
        data = self.equity_curve
        if last_n_points:
            data = data[-last_n_points:]

        return [
            {
                "timestamp": point.timestamp,
                "balance": point.balance,
                "pnl_change": point.pnl_change,
                "drawdown_pct": (self.peak_balance - point.balance) / self.initial_balance * 100
            }
            for point in data
        ]

    def reset_to_peak(self):
        """
        Reset tracking to current peak (useful for recovery scenarios)
        WARNING: This should only be used in specific recovery scenarios
        """
        if not self.drawdown_violated:
            self.current_balance = self.peak_balance
            self.current_drawdown = 0.0
            self.in_drawdown_period = False
            self.logger.warning("Drawdown monitor reset to peak balance")

    def simulate_loss(self, loss_amount: float) -> Dict:
        """
        Simulate the impact of a potential loss without updating actual balance

        Args:
            loss_amount: Amount of potential loss

        Returns:
            Dict with simulation results
        """
        simulated_balance = self.current_balance - loss_amount
        simulated_drawdown = (self.peak_balance - simulated_balance) / self.initial_balance

        would_violate = simulated_drawdown >= self.max_drawdown

        return {
            "current_balance": self.current_balance,
            "simulated_balance": simulated_balance,
            "loss_amount": loss_amount,
            "current_drawdown_pct": self.current_drawdown * 100,
            "simulated_drawdown_pct": simulated_drawdown * 100,
            "would_violate": would_violate,
            "margin_to_violation": (self.max_drawdown - simulated_drawdown) * self.initial_balance
        }

# Test the drawdown monitor
if __name__ == "__main__":
    # Initialize with $100,000 account
    monitor = FTMODrawdownMonitor(initial_balance=100000.0)

    print("FTMO Drawdown Monitor Test:")
    print("=" * 50)

    # Test balance updates simulating trading
    balances = [100000, 102000, 98000, 95000, 97000, 90000, 88000]

    for i, balance in enumerate(balances):
        result = monitor.update_balance(balance)
        print(f"Update {i+1}: Balance=${balance:,} | "
              f"Drawdown={result['current_drawdown']:.2f}% | "
              f"Status={result['status']}")

    # Test trade permission check
    trade_check = monitor.can_take_trade(3000)
    print(f"\nCan take trade with $3,000 risk: {trade_check['can_trade']}")
    print(f"Remaining capacity: ${trade_check['remaining_capacity']:.2f}")

    # Get comprehensive metrics
    metrics = monitor.get_drawdown_metrics()
    print(f"\nDrawdown Metrics:")
    print(f"  Max DD Reached: {metrics['max_drawdown_reached_pct']:.2f}%")
    print(f"  Current DD: {metrics['current_drawdown_pct']:.2f}%")
    print(f"  Status: {metrics['status']}")
    print(f"  Violation: {metrics['drawdown_violated']}")

    # Simulate a large loss
    simulation = monitor.simulate_loss(15000)
    print(f"\nSimulation - $15,000 loss:")
    print(f"  Would result in {simulation['simulated_drawdown_pct']:.2f}% drawdown")
    print(f"  Would violate limit: {simulation['would_violate']}")