import os
#!/usr/bin/env python3
"""
FTMO Currency Exposure Manager
Implements currency correlation tracking and exposure limits per FTMO requirements
"""
import logging
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import math

class ExposureStatus(Enum):
    SAFE = "SAFE"
    WARNING = "WARNING"
    LIMIT_REACHED = "LIMIT_REACHED"
    VIOLATION = "VIOLATION"

@dataclass
class Position:
    """Represents an open trading position"""
    trade_id: str
    symbol: str
    side: str  # "BUY" or "SELL"
    size_lots: float
    entry_price: float
    entry_time: datetime
    current_price: float = None
    unrealized_pnl: float = 0.0

@dataclass
class CurrencyExposure:
    """Tracks exposure for a single currency"""
    currency: str
    long_exposure: float = 0.0
    short_exposure: float = 0.0
    net_exposure: float = 0.0
    exposure_percentage: float = 0.0
    position_count: int = 0

class FTMOExposureManager:
    """
    Manage currency exposure and correlation limits per FTMO requirements
    """
    
    def __init__(self, account_balance: float, max_currency_exposure: float = 0.05,
                 max_correlated_exposure: float = 0.08):
        """
        Initialize exposure manager
        
        Args:
            account_balance: Current account balance
            max_currency_exposure: Maximum exposure per currency (default 5%)
            max_correlated_exposure: Maximum total exposure for correlated pairs (default 8%)
        """
        self.account_balance = account_balance
        self.max_currency_exposure = max_currency_exposure
        self.max_correlated_exposure = max_correlated_exposure
        
        # Position tracking
        self.open_positions: Dict[str, Position] = {}
        
        # Currency exposure tracking
        self.currency_exposures: Dict[str, CurrencyExposure] = defaultdict(
            lambda: CurrencyExposure(currency="")
        )
        
        # Major currency correlation matrix (simplified)
        self.correlation_matrix = {
            ("EURUSD", "GBPUSD"): 0.85,     # High positive correlation
            ("EURUSD", "USDCHF"): -0.92,    # High negative correlation
            ("EURUSD", "USDJPY"): -0.15,    # Low correlation
            ("EURUSD", "AUDUSD"): 0.72,     # Moderate positive correlation
            ("GBPUSD", "EURGBP"): -0.78,    # High negative correlation
            ("GBPUSD", "USDCHF"): -0.87,    # High negative correlation
            ("GBPUSD", "USDJPY"): -0.22,    # Low correlation
            ("AUDUSD", "NZDUSD"): 0.89,     # Very high positive correlation
            ("USDCAD", "USDJPY"): 0.45,     # Moderate positive correlation
            ("EURJPY", "GBPJPY"): 0.91,     # Very high positive correlation
            ("EURUSD", "EURGBP"): 0.35,     # Moderate positive correlation
            ("GBPUSD", "AUDUSD"): 0.68,     # Moderate positive correlation
        }
        
        # Highly correlated pairs groups
        self.correlation_groups = {
            "USD_MAJORS": ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"],
            "USD_SAFE_HAVENS": ["USDCHF", "USDJPY"],
            "EUR_CROSS": ["EURGBP", "EURJPY", "EURAUD", "EURCAD"],
            "GBP_CROSS": ["GBPJPY", "GBPAUD", "GBPCAD", "GBPCHF"],
            "COMMODITY_PAIRS": ["AUDUSD", "NZDUSD", "USDCAD"],
            "JPY_CROSS": ["EURJPY", "GBPJPY", "AUDJPY", "CADJPY"]
        }
        
        # Standard lot size
        self.standard_lot_size = 100000
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def check_new_position_allowed(self, symbol: str, side: str, size_lots: float,
                                 entry_price: float) -> Dict:
        """
        Check if a new position violates exposure limits
        
        Args:
            symbol: Currency pair (e.g., "EURUSD")
            side: Trade direction ("BUY" or "SELL")
            size_lots: Position size in lots
            entry_price: Entry price
            
        Returns:
            Dict with permission status and details
        """
        try:
            # Parse currencies from symbol
            base_currency = symbol[:3]
            quote_currency = symbol[3:]
            
            # Calculate position value
            position_value = size_lots * self.standard_lot_size * entry_price
            
            # Simulate adding the position
            simulated_exposures = self._simulate_position_addition(
                symbol, side, size_lots, position_value
            )
            
            # Check individual currency limits
            currency_violations = []
            for currency, exposure in simulated_exposures.items():
                if abs(exposure.exposure_percentage) > self.max_currency_exposure:
                    currency_violations.append({
                        "currency": currency,
                        "current_exposure": exposure.exposure_percentage * 100,
                        "limit": self.max_currency_exposure * 100
                    })
            
            # Check correlation group limits
            correlation_violations = self._check_correlation_violations(symbol, side, size_lots)
            
            # Check overall portfolio risk
            portfolio_risk = self._calculate_portfolio_risk(simulated_exposures)
            
            # Determine if position is allowed
            is_allowed = (len(currency_violations) == 0 and 
                         len(correlation_violations) == 0 and
                         portfolio_risk["total_exposure_pct"] <= 50.0)  # Max 50% total exposure (more realistic)
            
            # Determine status
            if not is_allowed:
                status = ExposureStatus.VIOLATION
            elif (max([abs(exp.exposure_percentage) for exp in simulated_exposures.values()], default=0) 
                  > self.max_currency_exposure * 0.8):
                status = ExposureStatus.WARNING
            else:
                status = ExposureStatus.SAFE
            
            result = {
                "allowed": is_allowed,
                "status": status.value,
                "symbol": symbol,
                "side": side,
                "size_lots": size_lots,
                "currency_violations": currency_violations,
                "correlation_violations": correlation_violations,
                "portfolio_risk": portfolio_risk,
                "simulated_exposures": {k: {
                    "currency": v.currency,
                    "net_exposure": v.net_exposure,
                    "exposure_percentage": v.exposure_percentage * 100
                } for k, v in simulated_exposures.items()},
                "recommendation": self._get_recommendation(is_allowed, currency_violations, correlation_violations)
            }
            
            self.logger.info(f"Position check for {symbol} {side} {size_lots} lots: "
                           f"{'ALLOWED' if is_allowed else 'REJECTED'} ({status.value})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Position check error: {e}")
            return {"allowed": False, "error": str(e)}
    
    def add_position(self, trade_id: str, symbol: str, side: str, size_lots: float,
                    entry_price: float, entry_time: datetime = None) -> Dict:
        """
        Add a new position to exposure tracking
        
        Args:
            trade_id: Unique trade identifier
            symbol: Currency pair
            side: Trade direction
            size_lots: Position size in lots
            entry_price: Entry price
            entry_time: Entry timestamp (default: now)
            
        Returns:
            Dict with position addition results
        """
        if entry_time is None:
            entry_time = datetime.utcnow()
        
        # Check if position is allowed first
        check_result = self.check_new_position_allowed(symbol, side, size_lots, entry_price)
        if not check_result["allowed"]:
            return {
                "success": False,
                "reason": "EXPOSURE_LIMIT_VIOLATED",
                "details": check_result
            }
        
        # Create position object
        position = Position(
            trade_id=trade_id,
            symbol=symbol,
            side=side,
            size_lots=size_lots,
            entry_price=entry_price,
            entry_time=entry_time,
            current_price=entry_price
        )
        
        # Add to tracking
        self.open_positions[trade_id] = position
        
        # Update exposures
        self._update_currency_exposures()
        
        self.logger.info(f"Position added: {trade_id} - {symbol} {side} {size_lots} lots @ {entry_price}")
        
        return {
            "success": True,
            "trade_id": trade_id,
            "position": position,
            "updated_exposures": self.get_current_exposures()
        }
    
    def remove_position(self, trade_id: str) -> Dict:
        """
        Remove position from exposure tracking
        
        Args:
            trade_id: Trade identifier to remove
            
        Returns:
            Dict with removal results
        """
        if trade_id not in self.open_positions:
            return {
                "success": False,
                "reason": "POSITION_NOT_FOUND",
                "trade_id": trade_id
            }
        
        # Get position details before removal
        position = self.open_positions[trade_id]
        
        # Remove position
        del self.open_positions[trade_id]
        
        # Update exposures
        self._update_currency_exposures()
        
        self.logger.info(f"Position removed: {trade_id} - {position.symbol}")
        
        return {
            "success": True,
            "trade_id": trade_id,
            "removed_position": position,
            "updated_exposures": self.get_current_exposures()
        }
    
    def update_position_price(self, trade_id: str, current_price: float) -> Dict:
        """
        Update current price for a position (for P&L tracking)
        
        Args:
            trade_id: Trade identifier
            current_price: Current market price
            
        Returns:
            Dict with update results
        """
        if trade_id not in self.open_positions:
            return {"success": False, "reason": "POSITION_NOT_FOUND"}
        
        position = self.open_positions[trade_id]
        position.current_price = current_price
        
        # Calculate unrealized P&L (simplified)
        pip_size = 0.01 if "JPY" in position.symbol else 0.0001
        pip_value = 10.0  # Simplified pip value
        
        if position.side == "BUY":
            price_diff_pips = (current_price - position.entry_price) / pip_size
        else:
            price_diff_pips = (position.entry_price - current_price) / pip_size
        
        position.unrealized_pnl = price_diff_pips * pip_value * position.size_lots
        
        return {
            "success": True,
            "trade_id": trade_id,
            "unrealized_pnl": position.unrealized_pnl
        }
    
    def _simulate_position_addition(self, symbol: str, side: str, size_lots: float,
                                  position_value: float) -> Dict[str, CurrencyExposure]:
        """
        Simulate adding a position to calculate new exposures
        """
        # Create copy of current exposures
        simulated_exposures = {}
        for currency, exposure in self.currency_exposures.items():
            simulated_exposures[currency] = CurrencyExposure(
                currency=currency,
                long_exposure=exposure.long_exposure,
                short_exposure=exposure.short_exposure,
                net_exposure=exposure.net_exposure,
                exposure_percentage=exposure.exposure_percentage,
                position_count=exposure.position_count
            )
        
        # Parse currencies
        base_currency = symbol[:3]
        quote_currency = symbol[3:]
        
        # Initialize currencies if not exist
        if base_currency not in simulated_exposures:
            simulated_exposures[base_currency] = CurrencyExposure(currency=base_currency)
        if quote_currency not in simulated_exposures:
            simulated_exposures[quote_currency] = CurrencyExposure(currency=quote_currency)
        
        # Add simulated position impact
        position_value_usd = position_value  # Assuming USD account
        
        if side == "BUY":
            # Buying base currency, selling quote currency
            simulated_exposures[base_currency].long_exposure += position_value_usd
            simulated_exposures[quote_currency].short_exposure += position_value_usd
        else:
            # Selling base currency, buying quote currency
            simulated_exposures[base_currency].short_exposure += position_value_usd
            simulated_exposures[quote_currency].long_exposure += position_value_usd
        
        # Recalculate net exposures and percentages
        for currency, exposure in simulated_exposures.items():
            exposure.net_exposure = exposure.long_exposure - exposure.short_exposure
            exposure.exposure_percentage = exposure.net_exposure / self.account_balance
            exposure.position_count += 1
        
        return simulated_exposures
    
    def _check_correlation_violations(self, symbol: str, side: str, size_lots: float) -> List[Dict]:
        """
        Check for correlation-based exposure violations
        """
        violations = []
        
        # Check correlation groups
        for group_name, symbols in self.correlation_groups.items():
            if symbol in symbols:
                # Calculate total exposure for this correlation group
                group_exposure = self._calculate_group_exposure(symbols, symbol, side, size_lots)
                
                if abs(group_exposure) > self.max_correlated_exposure * self.account_balance:
                    violations.append({
                        "group": group_name,
                        "current_exposure": group_exposure,
                        "limit": self.max_correlated_exposure * self.account_balance,
                        "symbols": symbols
                    })
        
        return violations
    
    def _calculate_group_exposure(self, group_symbols: List[str], new_symbol: str = None,
                                new_side: str = None, new_size: float = None) -> float:
        """
        Calculate total exposure for a correlation group
        """
        total_exposure = 0.0
        
        # Add existing positions
        for position in self.open_positions.values():
            if position.symbol in group_symbols:
                position_value = position.size_lots * self.standard_lot_size * position.entry_price
                
                # Consider the position's impact on USD exposure
                if position.symbol.endswith("USD"):
                    if position.side == "BUY":
                        total_exposure += position_value
                    else:
                        total_exposure -= position_value
                elif position.symbol.startswith("USD"):
                    if position.side == "BUY":
                        total_exposure -= position_value
                    else:
                        total_exposure += position_value
        
        # Add potential new position
        if new_symbol and new_symbol in group_symbols:
            new_position_value = new_size * self.standard_lot_size
            if new_symbol.endswith("USD"):
                if new_side == "BUY":
                    total_exposure += new_position_value
                else:
                    total_exposure -= new_position_value
            elif new_symbol.startswith("USD"):
                if new_side == "BUY":
                    total_exposure -= new_position_value
                else:
                    total_exposure += new_position_value
        
        return total_exposure
    
    def _calculate_portfolio_risk(self, exposures: Dict[str, CurrencyExposure]) -> Dict:
        """
        Calculate overall portfolio risk metrics
        """
        total_long_exposure = sum(exp.long_exposure for exp in exposures.values())
        total_short_exposure = sum(exp.short_exposure for exp in exposures.values())
        total_gross_exposure = total_long_exposure + total_short_exposure
        total_net_exposure = total_long_exposure - total_short_exposure
        
        return {
            "total_long_exposure": total_long_exposure,
            "total_short_exposure": total_short_exposure,
            "total_gross_exposure": total_gross_exposure,
            "total_net_exposure": total_net_exposure,
            "total_exposure_pct": (total_gross_exposure / self.account_balance) * 100,
            "net_exposure_pct": (abs(total_net_exposure) / self.account_balance) * 100,
            "leverage_ratio": total_gross_exposure / self.account_balance
        }
    
    def _update_currency_exposures(self):
        """
        Recalculate all currency exposures based on current positions
        """
        # Reset exposures
        self.currency_exposures.clear()
        
        # Calculate exposures from all positions
        for position in self.open_positions.values():
            base_currency = position.symbol[:3]
            quote_currency = position.symbol[3:]
            
            position_value = position.size_lots * self.standard_lot_size * position.entry_price
            
            # Initialize currency exposures if not exist
            if base_currency not in self.currency_exposures:
                self.currency_exposures[base_currency] = CurrencyExposure(currency=base_currency)
            if quote_currency not in self.currency_exposures:
                self.currency_exposures[quote_currency] = CurrencyExposure(currency=quote_currency)
            
            # Update exposures based on position direction
            if position.side == "BUY":
                # Long base currency, short quote currency
                self.currency_exposures[base_currency].long_exposure += position_value
                self.currency_exposures[quote_currency].short_exposure += position_value
            else:
                # Short base currency, long quote currency
                self.currency_exposures[base_currency].short_exposure += position_value
                self.currency_exposures[quote_currency].long_exposure += position_value
            
            # Update position counts
            self.currency_exposures[base_currency].position_count += 1
            self.currency_exposures[quote_currency].position_count += 1
        
        # Calculate net exposures and percentages
        for exposure in self.currency_exposures.values():
            exposure.net_exposure = exposure.long_exposure - exposure.short_exposure
            exposure.exposure_percentage = exposure.net_exposure / self.account_balance
    
    def _get_recommendation(self, is_allowed: bool, currency_violations: List,
                          correlation_violations: List) -> str:
        """
        Get recommendation based on violation analysis
        """
        if is_allowed:
            return "PROCEED"
        elif currency_violations:
            return f"REDUCE_EXPOSURE_FOR_{currency_violations[0]['currency']}"
        elif correlation_violations:
            return f"REDUCE_CORRELATED_EXPOSURE"
        else:
            return "GENERAL_RISK_REDUCTION_NEEDED"
    
    def get_current_exposures(self) -> Dict:
        """
        Get current exposure summary
        """
        exposures_dict = {}
        for currency, exposure in self.currency_exposures.items():
            exposures_dict[currency] = {
                "long_exposure": exposure.long_exposure,
                "short_exposure": exposure.short_exposure,
                "net_exposure": exposure.net_exposure,
                "exposure_percentage": exposure.exposure_percentage * 100,
                "position_count": exposure.position_count,
                "status": self._get_exposure_status(exposure.exposure_percentage)
            }
        
        portfolio_risk = self._calculate_portfolio_risk(self.currency_exposures)
        
        return {
            "currency_exposures": exposures_dict,
            "portfolio_risk": portfolio_risk,
            "total_positions": len(self.open_positions),
            "account_balance": self.account_balance
        }
    
    def _get_exposure_status(self, exposure_pct: float) -> str:
        """
        Get status for individual currency exposure
        """
        abs_exposure = abs(exposure_pct)
        if abs_exposure >= self.max_currency_exposure:
            return "VIOLATION"
        elif abs_exposure >= self.max_currency_exposure * 0.8:
            return "WARNING"
        elif abs_exposure >= self.max_currency_exposure * 0.5:
            return "MODERATE"
        else:
            return "SAFE"
    
    def get_correlation_analysis(self, symbol: str) -> Dict:
        """
        Get correlation analysis for a specific symbol
        """
        correlations = {}
        
        for (pair1, pair2), correlation in self.correlation_matrix.items():
            if symbol == pair1:
                correlations[pair2] = correlation
            elif symbol == pair2:
                correlations[pair1] = correlation
        
        # Find correlation group
        symbol_groups = []
        for group_name, symbols in self.correlation_groups.items():
            if symbol in symbols:
                symbol_groups.append(group_name)
        
        return {
            "symbol": symbol,
            "correlations": correlations,
            "correlation_groups": symbol_groups,
            "highly_correlated": [pair for pair, corr in correlations.items() if abs(corr) > 0.7]
        }
    
    def update_account_balance(self, new_balance: float):
        """
        Update account balance and recalculate exposure percentages
        """
        old_balance = self.account_balance
        self.account_balance = new_balance
        
        # Recalculate all exposure percentages
        for exposure in self.currency_exposures.values():
            exposure.exposure_percentage = exposure.net_exposure / new_balance
        
        self.logger.info(f"Account balance updated: ${old_balance:.2f} → ${new_balance:.2f}")

# Test the exposure manager
if __name__ == "__main__":
    # Initialize with $100,000 account
    manager = FTMOExposureManager(account_balance=100000.0)
    
    print("FTMO Exposure Manager Test:")
    print("=" * 50)
    
    # Test 1: Add EURUSD position (smaller size)
    result1 = manager.add_position("TRADE_001", "EURUSD", "BUY", 0.5, 1.0850)
    print(f"Add EURUSD position: {'SUCCESS' if result1['success'] else 'FAILED'}")
    
    # Test 2: Check adding correlated position
    check_result = manager.check_new_position_allowed("GBPUSD", "BUY", 0.3, 1.2500)
    print(f"GBPUSD position check: {'ALLOWED' if check_result['allowed'] else 'REJECTED'}")
    print(f"Status: {check_result['status']}")
    
    # Test 3: Add the GBPUSD position
    if check_result['allowed']:
        result2 = manager.add_position("TRADE_002", "GBPUSD", "BUY", 1.5, 1.2500)
        print(f"Add GBPUSD position: {'SUCCESS' if result2['success'] else 'FAILED'}")
    
    # Test 4: Get current exposures
    exposures = manager.get_current_exposures()
    print(f"\nCurrent Exposures:")
    for currency, data in exposures['currency_exposures'].items():
        if data['net_exposure'] != 0:
            print(f"  {currency}: {data['exposure_percentage']:.2f}% ({data['status']})")
    
    # Test 5: Portfolio risk
    print(f"\nPortfolio Risk:")
    print(f"  Total Exposure: {exposures['portfolio_risk']['total_exposure_pct']:.2f}%")
    print(f"  Leverage Ratio: {exposures['portfolio_risk']['leverage_ratio']:.2f}")
    
    # Test 6: Correlation analysis
    correlation = manager.get_correlation_analysis("EURUSD")
    print(f"\nEURUSD Correlations:")
    for pair, corr in correlation['correlations'].items():
        print(f"  {pair}: {corr:.2f}")
    print(f"  Groups: {correlation['correlation_groups']}")