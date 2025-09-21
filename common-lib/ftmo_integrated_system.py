import os
#!/usr/bin/env python3
"""
FTMO Integrated Trading System
Combines all FTMO compliance components into a unified trading system
"""
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date, timedelta
import json
from dataclasses import dataclass
from enum import Enum

# Import all our FTMO components
from ftmo_position_sizer import FTMOPositionSizer
from ftmo_daily_loss_monitor import FTMODailyLossMonitor
from ftmo_drawdown_monitor import FTMODrawdownMonitor
from ftmo_pnl_calculator import FTMOPnLCalculator
from ftmo_exposure_manager import FTMOExposureManager
from ftmo_market_filter import FTMOMarketFilter

class TradeDecision(Enum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    REDUCE_SIZE = "REDUCE_SIZE"
    DELAY = "DELAY"

@dataclass
class TradeRequest:
    """Represents a trade request to be validated"""
    symbol: str
    side: str  # "BUY" or "SELL"
    desired_size_lots: float
    entry_price: float
    stop_loss_price: float
    take_profit_price: Optional[float] = None
    timestamp: datetime = None
    trade_id: str = None

@dataclass
class TradeApproval:
    """Represents the system's decision on a trade request"""
    decision: TradeDecision
    approved_size_lots: float
    reasons: List[str]
    warnings: List[str]
    risk_metrics: Dict
    compliance_checks: Dict

class FTMOIntegratedSystem:
    """
    Integrated FTMO trading system with all compliance checks
    """

    def __init__(self, initial_balance: float = 100000.0):
        """
        Initialize the integrated FTMO system

        Args:
            initial_balance: Starting account balance
        """
        self.initial_balance = initial_balance
        self.current_balance = initial_balance

        # Initialize all subsystems
        self.position_sizer = FTMOPositionSizer(initial_balance)
        self.daily_monitor = FTMODailyLossMonitor(initial_balance)
        self.drawdown_monitor = FTMODrawdownMonitor(initial_balance)
        self.pnl_calculator = FTMOPnLCalculator()
        self.exposure_manager = FTMOExposureManager(initial_balance)
        self.market_filter = FTMOMarketFilter()

        # System state
        self.system_active = True
        self.risk_mode = "conservative"  # conservative, moderate, aggressive

        # Trade tracking
        self.pending_trades = {}
        self.executed_trades = {}
        self.rejected_trades = {}

        # Performance metrics
        self.system_metrics = {
            "trades_approved": 0,
            "trades_rejected": 0,
            "trades_executed": 0,
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "daily_violations": 0,
            "system_uptime": 100.0
        }

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"FTMO Integrated System initialized with ${initial_balance:,.2f}")

    def evaluate_trade_request(self, trade_request: TradeRequest) -> TradeApproval:
        """
        Comprehensive trade evaluation using all FTMO compliance systems

        Args:
            trade_request: Trade request to evaluate

        Returns:
            TradeApproval with decision and details
        """
        if trade_request.timestamp is None:
            trade_request.timestamp = datetime.utcnow()

        if trade_request.trade_id is None:
            trade_request.trade_id = f"TRADE_{len(self.pending_trades) + 1:04d}"

        self.logger.info(f"Evaluating trade request: {trade_request.trade_id} - "
                        f"{trade_request.symbol} {trade_request.side} {trade_request.desired_size_lots} lots")

        # Initialize approval structure
        approval = TradeApproval(
            decision=TradeDecision.APPROVE,
            approved_size_lots=trade_request.desired_size_lots,
            reasons=[],
            warnings=[],
            risk_metrics={},
            compliance_checks={}
        )

        # Check 1: System status
        if not self.system_active:
            approval.decision = TradeDecision.REJECT
            approval.reasons.append("SYSTEM_INACTIVE")
            return approval

        # Check 2: Market conditions
        market_check = self.market_filter.is_trading_allowed(
            trade_request.timestamp, trade_request.symbol, self.risk_mode
        )
        approval.compliance_checks["market"] = market_check

        if not market_check["allowed"]:
            approval.decision = TradeDecision.REJECT
            approval.reasons.extend(market_check["reasons"])
            return approval
        elif market_check["recommendation"].value == "REDUCE_SIZE":
            approval.decision = TradeDecision.REDUCE_SIZE
            approval.warnings.extend(market_check["warnings"])

        # Check 3: Position sizing
        position_check = self.position_sizer.calculate_position_size(
            trade_request.entry_price,
            trade_request.stop_loss_price,
            trade_request.symbol,
            trade_request.side
        )
        approval.compliance_checks["position_sizing"] = position_check

        if not position_check["is_valid"]:
            approval.decision = TradeDecision.REJECT
            approval.reasons.append("INVALID_POSITION_SIZE")
            return approval

        # Use system-recommended size if smaller than requested
        system_recommended_size = position_check["position_size_lots"]
        if system_recommended_size < trade_request.desired_size_lots:
            approval.approved_size_lots = system_recommended_size
            approval.decision = TradeDecision.REDUCE_SIZE
            approval.warnings.append(f"SIZE_REDUCED_TO_{system_recommended_size:.3f}_LOTS")

        # Check 4: Daily loss limits
        potential_loss = position_check["risk_amount"]
        daily_check = self.daily_monitor.can_take_trade(
            potential_loss, trade_request.timestamp.date()
        )
        approval.compliance_checks["daily_loss"] = daily_check

        if not daily_check["can_trade"]:
            approval.decision = TradeDecision.REJECT
            approval.reasons.append(daily_check["reason"])
            return approval

        # Check 5: Total drawdown limits
        drawdown_check = self.drawdown_monitor.can_take_trade(potential_loss)
        approval.compliance_checks["drawdown"] = drawdown_check

        if not drawdown_check["can_trade"]:
            approval.decision = TradeDecision.REJECT
            approval.reasons.append(drawdown_check["reason"])
            return approval
        elif not drawdown_check["is_safe"]:
            approval.warnings.append("APPROACHING_DRAWDOWN_LIMIT")

        # Check 6: Currency exposure limits
        exposure_check = self.exposure_manager.check_new_position_allowed(
            trade_request.symbol,
            trade_request.side,
            approval.approved_size_lots,
            trade_request.entry_price
        )
        approval.compliance_checks["exposure"] = exposure_check

        if not exposure_check["allowed"]:
            approval.decision = TradeDecision.REJECT
            approval.reasons.append("EXPOSURE_LIMIT_VIOLATED")
            approval.reasons.extend([v["currency"] for v in exposure_check.get("currency_violations", [])])
            return approval

        # Check 7: Calculate realistic P&L expectations
        breakeven_analysis = self.pnl_calculator.calculate_breakeven_move(
            trade_request.symbol,
            trade_request.side,
            approval.approved_size_lots,
            trade_request.entry_price
        )
        approval.risk_metrics["breakeven_analysis"] = breakeven_analysis

        # Final decision logic
        if len(approval.warnings) >= 3:
            approval.decision = TradeDecision.DELAY
            approval.reasons.append("MULTIPLE_WARNINGS")

        # Calculate risk metrics
        approval.risk_metrics.update({
            "position_size_lots": approval.approved_size_lots,
            "risk_amount": potential_loss,
            "risk_percentage": (potential_loss / self.current_balance) * 100,
            "remaining_daily_capacity": daily_check["remaining_capacity"],
            "remaining_drawdown_capacity": drawdown_check["remaining_capacity"],
            "breakeven_pips": breakeven_analysis["breakeven_move_pips"]
        })

        # Update metrics
        if approval.decision == TradeDecision.APPROVE:
            self.system_metrics["trades_approved"] += 1
        else:
            self.system_metrics["trades_rejected"] += 1

        self.logger.info(f"Trade evaluation complete: {trade_request.trade_id} - "
                        f"Decision: {approval.decision.value}, "
                        f"Size: {approval.approved_size_lots:.3f} lots")

        return approval

    def execute_trade(self, trade_request: TradeRequest, approval: TradeApproval) -> Dict:
        """
        Execute an approved trade and update all systems

        Args:
            trade_request: Original trade request
            approval: Approved trade details

        Returns:
            Dict with execution results
        """
        if approval.decision not in [TradeDecision.APPROVE, TradeDecision.REDUCE_SIZE]:
            return {
                "success": False,
                "reason": "TRADE_NOT_APPROVED",
                "decision": approval.decision.value
            }

        try:
            # Add position to exposure tracking
            exposure_result = self.exposure_manager.add_position(
                trade_request.trade_id,
                trade_request.symbol,
                trade_request.side,
                approval.approved_size_lots,
                trade_request.entry_price,
                trade_request.timestamp
            )

            if not exposure_result["success"]:
                return {
                    "success": False,
                    "reason": "EXPOSURE_TRACKING_FAILED",
                    "details": exposure_result
                }

            # Record trade execution
            executed_trade = {
                "trade_id": trade_request.trade_id,
                "symbol": trade_request.symbol,
                "side": trade_request.side,
                "size_lots": approval.approved_size_lots,
                "entry_price": trade_request.entry_price,
                "stop_loss": trade_request.stop_loss_price,
                "take_profit": trade_request.take_profit_price,
                "entry_time": trade_request.timestamp,
                "status": "OPEN",
                "approval": approval,
                "exposure_tracking": exposure_result
            }

            self.executed_trades[trade_request.trade_id] = executed_trade
            self.system_metrics["trades_executed"] += 1

            self.logger.info(f"Trade executed successfully: {trade_request.trade_id}")

            return {
                "success": True,
                "trade_id": trade_request.trade_id,
                "executed_trade": executed_trade
            }

        except Exception as e:
            self.logger.error(f"Trade execution failed: {e}")
            return {
                "success": False,
                "reason": "EXECUTION_ERROR",
                "error": str(e)
            }

    def close_trade(self, trade_id: str, exit_price: float, exit_time: datetime = None) -> Dict:
        """
        Close an open trade and update all systems

        Args:
            trade_id: Trade identifier
            exit_price: Exit price
            exit_time: Exit timestamp (default: now)

        Returns:
            Dict with closure results
        """
        if exit_time is None:
            exit_time = datetime.utcnow()

        if trade_id not in self.executed_trades:
            return {
                "success": False,
                "reason": "TRADE_NOT_FOUND"
            }

        trade = self.executed_trades[trade_id]

        if trade["status"] != "OPEN":
            return {
                "success": False,
                "reason": "TRADE_ALREADY_CLOSED"
            }

        try:
            # Calculate P&L
            pnl_result = self.pnl_calculator.calculate_trade_pnl(
                trade["symbol"],
                trade["side"],
                trade["entry_price"],
                exit_price,
                trade["size_lots"],
                trade["entry_time"],
                exit_time
            )

            # Update daily loss monitoring
            daily_update = self.daily_monitor.update_daily_pnl(
                pnl_result["net_pnl"],
                exit_time.date(),
                trade_id,
                trade["symbol"]
            )

            # Update account balance and drawdown monitoring
            new_balance = self.current_balance + pnl_result["net_pnl"]
            drawdown_update = self.drawdown_monitor.update_balance(new_balance, exit_time)
            self.current_balance = new_balance

            # Update all systems with new balance
            self.position_sizer.update_account_balance(new_balance)
            self.daily_monitor.update_account_balance(new_balance)
            self.exposure_manager.update_account_balance(new_balance)

            # Remove from exposure tracking
            exposure_removal = self.exposure_manager.remove_position(trade_id)

            # Update trade record
            trade.update({
                "status": "CLOSED",
                "exit_price": exit_price,
                "exit_time": exit_time,
                "pnl_breakdown": pnl_result,
                "net_pnl": pnl_result["net_pnl"],
                "daily_update": daily_update,
                "drawdown_update": drawdown_update
            })

            # Update system metrics
            self.system_metrics["total_pnl"] += pnl_result["net_pnl"]
            if pnl_result["net_pnl"] > 0:
                self.system_metrics["wins"] = self.system_metrics.get("wins", 0) + 1
            else:
                self.system_metrics["losses"] = self.system_metrics.get("losses", 0) + 1

            total_closed = self.system_metrics.get("wins", 0) + self.system_metrics.get("losses", 0)
            if total_closed > 0:
                self.system_metrics["win_rate"] = (self.system_metrics.get("wins", 0) / total_closed) * 100

            self.logger.info(f"Trade closed: {trade_id} - P&L: ${pnl_result['net_pnl']:.2f}")

            return {
                "success": True,
                "trade_id": trade_id,
                "pnl": pnl_result["net_pnl"],
                "new_balance": new_balance,
                "compliance_status": {
                    "daily_loss_ok": not daily_update["trading_suspended"],
                    "drawdown_ok": not drawdown_update["drawdown_violated"]
                }
            }

        except Exception as e:
            self.logger.error(f"Trade closure failed: {e}")
            return {
                "success": False,
                "reason": "CLOSURE_ERROR",
                "error": str(e)
            }

    def get_system_status(self) -> Dict:
        """
        Get comprehensive system status
        """
        return {
            "system_active": self.system_active,
            "current_balance": self.current_balance,
            "initial_balance": self.initial_balance,
            "total_return": ((self.current_balance - self.initial_balance) / self.initial_balance) * 100,
            "risk_mode": self.risk_mode,
            "open_trades": len([t for t in self.executed_trades.values() if t["status"] == "OPEN"]),
            "total_trades": len(self.executed_trades),
            "pending_trades": len(self.pending_trades),
            "daily_loss_status": self.daily_monitor.get_daily_summary(),
            "drawdown_status": self.drawdown_monitor.get_drawdown_metrics(),
            "exposure_status": self.exposure_manager.get_current_exposures(),
            "system_metrics": self.system_metrics,
            "compliance_summary": self._get_compliance_summary()
        }

    def _get_compliance_summary(self) -> Dict:
        """
        Get FTMO compliance summary
        """
        daily_summary = self.daily_monitor.get_daily_summary()
        drawdown_metrics = self.drawdown_monitor.get_drawdown_metrics()

        return {
            "daily_loss_compliant": daily_summary["daily_loss_percentage"] < 5.0,
            "total_drawdown_compliant": not drawdown_metrics["drawdown_violated"],
            "position_sizing_compliant": True,  # Enforced by system
            "exposure_compliant": True,  # Enforced by system
            "trading_days": len(self.daily_monitor.daily_trades),
            "profit_target_progress": ((self.current_balance - self.initial_balance) / self.initial_balance) * 100,
            "violations": {
                "daily_loss": len(self.daily_monitor.violation_history),
                "drawdown": 1 if drawdown_metrics["drawdown_violated"] else 0,
                "total": len(self.daily_monitor.violation_history) + (1 if drawdown_metrics["drawdown_violated"] else 0)
            }
        }

    def set_risk_mode(self, mode: str):
        """
        Set system risk mode (conservative, moderate, aggressive)
        """
        if mode in ["conservative", "moderate", "aggressive"]:
            self.risk_mode = mode
            self.logger.info(f"Risk mode set to: {mode}")
        else:
            raise ValueError("Risk mode must be 'conservative', 'moderate', or 'aggressive'")

    def emergency_stop(self, reason: str = "MANUAL_STOP"):
        """
        Emergency stop all trading
        """
        self.system_active = False
        self.logger.critical(f"EMERGENCY STOP ACTIVATED: {reason}")

    def resume_trading(self):
        """
        Resume trading after emergency stop
        """
        self.system_active = True
        self.logger.info("Trading resumed")

# Test the integrated system
if __name__ == "__main__":
    # Initialize the integrated system
    system = FTMOIntegratedSystem(initial_balance=100000.0)

    print("FTMO Integrated System Test:")
    print("=" * 60)

    # Test 1: Create a trade request
    trade_request = TradeRequest(
        symbol="EURUSD",
        side="BUY",
        desired_size_lots=0.1,  # Small position for testing
        entry_price=1.0850,
        stop_loss_price=1.0800,  # 50 pip stop loss
        take_profit_price=1.0900  # 50 pip take profit
    )

    print(f"Test Trade Request:")
    print(f"  Symbol: {trade_request.symbol}")
    print(f"  Direction: {trade_request.side}")
    print(f"  Size: {trade_request.desired_size_lots} lots")
    print(f"  Entry: {trade_request.entry_price}")
    print(f"  Stop Loss: {trade_request.stop_loss_price}")

    # Test 2: Evaluate the trade
    approval = system.evaluate_trade_request(trade_request)
    print(f"\nTrade Evaluation Result:")
    print(f"  Decision: {approval.decision.value}")
    print(f"  Approved Size: {approval.approved_size_lots:.3f} lots")
    print(f"  Reasons: {approval.reasons}")
    print(f"  Warnings: {approval.warnings}")
    print(f"  Risk Amount: ${approval.risk_metrics.get('risk_amount', 0):.2f}")

    # Test 3: Execute if approved
    if approval.decision in [TradeDecision.APPROVE, TradeDecision.REDUCE_SIZE]:
        execution_result = system.execute_trade(trade_request, approval)
        print(f"\nTrade Execution:")
        print(f"  Success: {execution_result['success']}")
        if execution_result['success']:
            print(f"  Trade ID: {execution_result['trade_id']}")
        else:
            print(f"  Reason: {execution_result['reason']}")

    # Test 4: Get system status
    status = system.get_system_status()
    print(f"\nSystem Status:")
    print(f"  Balance: ${status['current_balance']:,.2f}")
    print(f"  Open Trades: {status['open_trades']}")
    print(f"  Total Trades: {status['total_trades']}")
    print(f"  Daily Loss Compliant: {status['compliance_summary']['daily_loss_compliant']}")
    print(f"  Drawdown Compliant: {status['compliance_summary']['total_drawdown_compliant']}")

    print(f"\n✅ FTMO Integrated System Test Complete!")