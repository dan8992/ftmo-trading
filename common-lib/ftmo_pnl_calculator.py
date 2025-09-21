import os
#!/usr/bin/env python3
"""
FTMO P&L Calculator with Realistic Transaction Costs
Implements accurate P&L calculations including spreads, commissions, and swap fees
"""
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, time
from enum import Enum
import math

class TradeDirection(Enum):
    BUY = "BUY"
    SELL = "SELL"

class FTMOPnLCalculator:
    """
    Calculate realistic P&L for FTMO trading including all transaction costs
    """
    
    def __init__(self):
        """
        Initialize P&L calculator with market-realistic costs
        """
        # Spread costs in pips (typical FTMO broker spreads)
        self.spread_costs = {
            "EURUSD": 0.7,   # 0.7 pips
            "GBPUSD": 1.2,   # 1.2 pips
            "USDJPY": 0.8,   # 0.8 pips
            "USDCHF": 1.5,   # 1.5 pips
            "USDCAD": 1.8,   # 1.8 pips
            "AUDUSD": 1.5,   # 1.5 pips
            "NZDUSD": 2.0,   # 2.0 pips
            "EURJPY": 1.5,   # 1.5 pips
            "GBPJPY": 2.5,   # 2.5 pips
            "EURGBP": 1.0,   # 1.0 pips
            "AUDCAD": 2.2,   # 2.2 pips
            "AUDCHF": 2.5,   # 2.5 pips
            "AUDJPY": 2.0,   # 2.0 pips
            "CADCHF": 2.8,   # 2.8 pips
            "CADJPY": 2.2,   # 2.2 pips
            "CHFJPY": 2.5,   # 2.5 pips
            "EURAUD": 2.8,   # 2.8 pips
            "EURCAD": 2.5,   # 2.5 pips
            "EURCHF": 1.8,   # 1.8 pips
            "GBPAUD": 3.5,   # 3.5 pips
            "GBPCAD": 3.8,   # 3.8 pips
            "GBPCHF": 2.8,   # 2.8 pips
            "NZDCAD": 3.0,   # 3.0 pips
            "NZDCHF": 3.5,   # 3.5 pips
            "NZDJPY": 2.8,   # 2.8 pips
        }
        
        # Commission rates (most FTMO brokers are spread-only, but some charge commissions)
        self.commission_rates = {
            "default": 0.0,  # No commission for most FTMO brokers
            # Some brokers might charge: "EURUSD": 3.5  # $3.5 per lot per side
        }
        
        # Swap rates (overnight financing costs) - approximate rates
        self.swap_rates_long = {
            "EURUSD": -0.8,   # Daily swap for long position
            "GBPUSD": -1.2,
            "USDJPY": 0.5,
            "USDCHF": 0.8,
            "AUDUSD": -0.5,
            # Add more as needed
        }
        
        self.swap_rates_short = {
            "EURUSD": 0.2,    # Daily swap for short position
            "GBPUSD": 0.5,
            "USDJPY": -2.5,
            "USDCHF": -3.0,
            "AUDUSD": -0.8,
            # Add more as needed
        }
        
        # Standard contract specifications
        self.contract_sizes = {
            "FOREX": 100000,  # Standard lot = 100,000 units
        }
        
        # Slippage estimates by market conditions
        self.slippage_estimates = {
            "normal": 0.2,      # 0.2 pips average slippage
            "volatile": 0.8,    # 0.8 pips during volatile periods
            "news": 2.0,        # 2.0 pips during news events
            "illiquid": 1.5,    # 1.5 pips during illiquid hours
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def calculate_trade_pnl(self, symbol: str, side: str, entry_price: float, 
                           exit_price: float, position_size_lots: float,
                           entry_time: datetime = None, exit_time: datetime = None,
                           market_condition: str = "normal") -> Dict:
        """
        Calculate comprehensive P&L including all costs
        
        Args:
            symbol: Currency pair (e.g., "EURUSD")
            side: Trade direction ("BUY" or "SELL")
            entry_price: Entry price
            exit_price: Exit price
            position_size_lots: Position size in lots
            entry_time: Entry timestamp (for swap calculation)
            exit_time: Exit timestamp (for swap calculation)
            market_condition: Market condition for slippage ("normal", "volatile", "news", "illiquid")
            
        Returns:
            Dict with detailed P&L breakdown
        """
        try:
            # Validate inputs
            if position_size_lots <= 0:
                raise ValueError("Position size must be positive")
            if entry_price <= 0 or exit_price <= 0:
                raise ValueError("Prices must be positive")
            if symbol not in self.spread_costs:
                raise ValueError(f"Unsupported symbol: {symbol}")
            
            # Calculate base P&L
            pip_size = self._get_pip_size(symbol)
            pip_value = self._get_pip_value(symbol, entry_price)
            
            # Calculate price difference in pips
            if side.upper() == TradeDirection.BUY.value:
                price_diff_pips = (exit_price - entry_price) / pip_size
            else:  # SELL
                price_diff_pips = (entry_price - exit_price) / pip_size
            
            # Gross P&L before costs
            gross_pnl = price_diff_pips * pip_value * position_size_lots
            
            # Calculate transaction costs
            costs = self._calculate_all_costs(
                symbol, side, position_size_lots, pip_value,
                entry_time, exit_time, market_condition
            )
            
            # Net P&L after all costs
            net_pnl = gross_pnl - costs["total_cost"]
            
            # Calculate return metrics
            notional_value = position_size_lots * self.contract_sizes["FOREX"] * entry_price
            return_on_notional = (net_pnl / notional_value) * 100 if notional_value > 0 else 0
            
            result = {
                "symbol": symbol,
                "side": side,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "position_size_lots": position_size_lots,
                "price_diff_pips": price_diff_pips,
                "pip_value": pip_value,
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl,
                "notional_value": notional_value,
                "return_on_notional_pct": return_on_notional,
                "costs_breakdown": costs,
                "market_condition": market_condition,
                "calculated_at": datetime.utcnow()
            }
            
            self.logger.info(f"P&L calculated for {symbol} {side}: "
                           f"Gross ${gross_pnl:.2f}, Net ${net_pnl:.2f}, "
                           f"Costs ${costs['total_cost']:.2f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"P&L calculation error: {e}")
            return self._get_error_result(symbol, str(e))
    
    def _calculate_all_costs(self, symbol: str, side: str, position_size_lots: float,
                           pip_value: float, entry_time: datetime = None,
                           exit_time: datetime = None, market_condition: str = "normal") -> Dict:
        """
        Calculate all transaction costs
        """
        # Spread cost (always applies)
        spread_pips = self.spread_costs.get(symbol, 1.0)
        spread_cost = spread_pips * pip_value * position_size_lots
        
        # Commission cost
        commission = self._calculate_commission(symbol, position_size_lots)
        
        # Slippage cost
        slippage_cost = self._calculate_slippage(symbol, position_size_lots, pip_value, market_condition)
        
        # Swap cost (if position held overnight)
        swap_cost = self._calculate_swap_cost(symbol, side, position_size_lots, pip_value, entry_time, exit_time)
        
        # Total cost
        total_cost = spread_cost + commission + slippage_cost + abs(swap_cost)
        
        return {
            "spread_cost": spread_cost,
            "spread_pips": spread_pips,
            "commission": commission,
            "slippage_cost": slippage_cost,
            "swap_cost": swap_cost,  # Can be positive (credit) or negative (debit)
            "total_cost": total_cost
        }
    
    def _calculate_commission(self, symbol: str, position_size_lots: float) -> float:
        """
        Calculate commission costs
        """
        commission_rate = self.commission_rates.get(symbol, self.commission_rates["default"])
        return commission_rate * position_size_lots * 2  # Round trip (entry + exit)
    
    def _calculate_slippage(self, symbol: str, position_size_lots: float, 
                          pip_value: float, market_condition: str) -> float:
        """
        Calculate estimated slippage costs
        """
        slippage_pips = self.slippage_estimates.get(market_condition, 0.5)
        
        # Larger positions may have higher slippage
        size_multiplier = 1.0 + (position_size_lots - 1.0) * 0.1  # 10% more slippage per additional lot
        adjusted_slippage = slippage_pips * size_multiplier
        
        return adjusted_slippage * pip_value * position_size_lots * 2  # Round trip
    
    def _calculate_swap_cost(self, symbol: str, side: str, position_size_lots: float,
                           pip_value: float, entry_time: datetime = None,
                           exit_time: datetime = None) -> float:
        """
        Calculate swap (overnight financing) costs
        """
        if not entry_time or not exit_time:
            return 0.0  # Can't calculate without timestamps
        
        # Check if position was held overnight
        days_held = (exit_time.date() - entry_time.date()).days
        if days_held <= 0:
            return 0.0  # Intraday trade, no swap
        
        # Get swap rate for the symbol and direction
        if side.upper() == TradeDirection.BUY.value:
            swap_rate = self.swap_rates_long.get(symbol, 0.0)
        else:
            swap_rate = self.swap_rates_short.get(symbol, 0.0)
        
        # Calculate swap cost (negative = cost, positive = credit)
        swap_cost = swap_rate * position_size_lots * days_held
        
        # Account for Wednesday triple swap (rollover for weekend)
        if self._is_wednesday_rollover(entry_time, exit_time):
            swap_cost *= 3
        
        return swap_cost
    
    def _is_wednesday_rollover(self, entry_time: datetime, exit_time: datetime) -> bool:
        """
        Check if position was held over Wednesday (which includes weekend rollover)
        """
        # Check if any Wednesday fell between entry and exit
        current_date = entry_time.date()
        end_date = exit_time.date()
        
        while current_date <= end_date:
            if current_date.weekday() == 2:  # Wednesday = 2
                return True
            current_date += datetime.timedelta(days=1)
        
        return False
    
    def _get_pip_value(self, symbol: str, price: float) -> float:
        """
        Calculate pip value in USD for different currency pairs
        """
        if symbol.endswith("USD"):
            # USD is quote currency
            if "JPY" in symbol:
                return 1000.0 / price  # For USDJPY
            else:
                return 10.0  # For EURUSD, GBPUSD, etc.
        elif symbol.startswith("USD"):
            # USD is base currency
            if symbol.endswith("JPY"):
                return 10.0  # For USDJPY (already handled above)
            else:
                return 10.0 / price  # For USDCHF, USDCAD
        elif symbol.endswith("JPY"):
            # JPY is quote currency (cross pairs)
            usd_jpy_rate = 150.0  # Approximate USD/JPY rate (should be fetched from market)
            return 1000.0 / usd_jpy_rate
        else:
            # Cross pairs without JPY or USD
            return 10.0  # Simplified calculation (should use actual rates)
    
    def _get_pip_size(self, symbol: str) -> float:
        """
        Get pip size for different currency pairs
        """
        if "JPY" in symbol:
            return 0.01  # Japanese Yen pairs use 2 decimal places
        else:
            return 0.0001  # Most major pairs use 4 decimal places
    
    def _get_error_result(self, symbol: str, error_msg: str) -> Dict:
        """
        Return error result structure
        """
        return {
            "symbol": symbol,
            "gross_pnl": 0.0,
            "net_pnl": 0.0,
            "error": error_msg,
            "calculated_at": datetime.utcnow()
        }
    
    def calculate_breakeven_move(self, symbol: str, side: str, position_size_lots: float,
                               entry_price: float, market_condition: str = "normal") -> Dict:
        """
        Calculate the minimum price move needed to break even after costs
        
        Args:
            symbol: Currency pair
            side: Trade direction
            position_size_lots: Position size
            entry_price: Entry price
            market_condition: Market condition for cost calculation
            
        Returns:
            Dict with breakeven analysis
        """
        pip_value = self._get_pip_value(symbol, entry_price)
        pip_size = self._get_pip_size(symbol)
        
        # Calculate total costs for a round trip
        costs = self._calculate_all_costs(symbol, side, position_size_lots, pip_value, 
                                        market_condition=market_condition)
        
        # Calculate breakeven move in pips
        breakeven_pips = costs["total_cost"] / (pip_value * position_size_lots)
        
        # Calculate breakeven price
        if side.upper() == TradeDirection.BUY.value:
            breakeven_price = entry_price + (breakeven_pips * pip_size)
        else:
            breakeven_price = entry_price - (breakeven_pips * pip_size)
        
        return {
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "breakeven_price": breakeven_price,
            "breakeven_move_pips": breakeven_pips,
            "total_costs": costs["total_cost"],
            "costs_breakdown": costs,
            "position_size_lots": position_size_lots
        }
    
    def get_symbol_cost_summary(self, symbol: str) -> Dict:
        """
        Get cost summary for a specific symbol
        """
        return {
            "symbol": symbol,
            "spread_pips": self.spread_costs.get(symbol, "Unknown"),
            "commission_per_lot": self.commission_rates.get(symbol, self.commission_rates["default"]),
            "swap_long": self.swap_rates_long.get(symbol, "Unknown"),
            "swap_short": self.swap_rates_short.get(symbol, "Unknown"),
            "pip_size": self._get_pip_size(symbol)
        }

# Test the P&L calculator
if __name__ == "__main__":
    calculator = FTMOPnLCalculator()
    
    print("FTMO P&L Calculator Test:")
    print("=" * 50)
    
    # Test 1: Profitable EURUSD trade
    result1 = calculator.calculate_trade_pnl(
        symbol="EURUSD",
        side="BUY",
        entry_price=1.0850,
        exit_price=1.0900,  # 50 pip profit
        position_size_lots=1.0,
        market_condition="normal"
    )
    
    print(f"Test 1 - EURUSD BUY (50 pips profit):")
    print(f"  Gross P&L: ${result1['gross_pnl']:.2f}")
    print(f"  Net P&L: ${result1['net_pnl']:.2f}")
    print(f"  Total Costs: ${result1['costs_breakdown']['total_cost']:.2f}")
    print(f"  Spread Cost: ${result1['costs_breakdown']['spread_cost']:.2f}")
    
    # Test 2: Losing trade with costs
    result2 = calculator.calculate_trade_pnl(
        symbol="GBPUSD",
        side="SELL",
        entry_price=1.2500,
        exit_price=1.2530,  # 30 pip loss
        position_size_lots=0.5,
        market_condition="volatile"
    )
    
    print(f"\nTest 2 - GBPUSD SELL (30 pips loss):")
    print(f"  Gross P&L: ${result2['gross_pnl']:.2f}")
    print(f"  Net P&L: ${result2['net_pnl']:.2f}")
    print(f"  Total Costs: ${result2['costs_breakdown']['total_cost']:.2f}")
    
    # Test 3: Breakeven analysis
    breakeven = calculator.calculate_breakeven_move(
        symbol="EURUSD",
        side="BUY",
        position_size_lots=1.0,
        entry_price=1.0850
    )
    
    print(f"\nBreakeven Analysis - EURUSD 1.0 lot:")
    print(f"  Entry Price: {breakeven['entry_price']:.5f}")
    print(f"  Breakeven Price: {breakeven['breakeven_price']:.5f}")
    print(f"  Required Move: {breakeven['breakeven_move_pips']:.1f} pips")
    
    # Test 4: Cost summary
    cost_summary = calculator.get_symbol_cost_summary("EURUSD")
    print(f"\nEURUSD Cost Summary:")
    print(f"  Spread: {cost_summary['spread_pips']} pips")
    print(f"  Commission: ${cost_summary['commission_per_lot']} per lot")
    print(f"  Swap Long: {cost_summary['swap_long']}")
    print(f"  Swap Short: {cost_summary['swap_short']}")