import os
#!/usr/bin/env python3
"""
FTMO-Compliant Position Sizing Engine
Implements realistic position sizing based on FTMO rules and risk management
"""
import logging
from typing import Dict, Tuple
from datetime import datetime

class FTMOPositionSizer:
    """
    Position sizing engine that complies with FTMO challenge requirements
    """
    
    def __init__(self, account_balance: float, max_risk_per_trade: float = 0.02):
        """
        Initialize position sizer
        
        Args:
            account_balance: Current account balance in USD
            max_risk_per_trade: Maximum risk per trade (default 2% for FTMO)
        """
        self.account_balance = account_balance
        self.max_risk_per_trade = max_risk_per_trade
        
        # FTMO position limits by symbol (in lots)
        self.max_position_limits = {
            "EURUSD": 2.0,
            "GBPUSD": 2.0,
            "USDJPY": 2.0,
            "AUDUSD": 2.0,
            "USDCAD": 2.0,
            "USDCHF": 2.0,
            "NZDUSD": 2.0,
            "EURJPY": 1.5,
            "GBPJPY": 1.5,
            "EURGBP": 1.5,
            # Add other pairs as needed
        }
        
        # Standard contract sizes
        self.contract_sizes = {
            "FOREX": 100000,  # Standard lot size
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def calculate_position_size(self, entry_price: float, stop_loss_price: float, 
                              symbol: str, side: str = "BUY") -> Dict:
        """
        Calculate FTMO-compliant position size
        
        Args:
            entry_price: Entry price for the trade
            stop_loss_price: Stop loss price
            symbol: Currency pair (e.g., "EURUSD")
            side: Trade direction ("BUY" or "SELL")
            
        Returns:
            Dict containing position size details
        """
        try:
            # Validate inputs
            if entry_price <= 0 or stop_loss_price <= 0:
                raise ValueError("Prices must be positive")
                
            if symbol not in self.max_position_limits:
                raise ValueError(f"Unsupported symbol: {symbol}")
            
            # Calculate risk amount in account currency
            risk_amount = self.account_balance * self.max_risk_per_trade
            
            # Calculate pip value and stop loss in pips
            pip_value = self._get_pip_value(symbol, entry_price)
            pip_size = self._get_pip_size(symbol)
            
            # Calculate stop loss distance in pips
            if side.upper() == "BUY":
                stop_loss_pips = (entry_price - stop_loss_price) / pip_size
            else:  # SELL
                stop_loss_pips = (stop_loss_price - entry_price) / pip_size
                
            if stop_loss_pips <= 0:
                raise ValueError("Stop loss must be in correct direction")
            
            # Calculate position size in lots
            risk_per_pip = stop_loss_pips * pip_value
            position_size_lots = risk_amount / risk_per_pip if risk_per_pip > 0 else 0
            
            # Apply FTMO maximum position limits
            max_lots = self.max_position_limits.get(symbol, 1.0)
            final_position_size = min(position_size_lots, max_lots)
            
            # Calculate actual risk with final position size
            actual_risk = final_position_size * risk_per_pip
            actual_risk_pct = actual_risk / self.account_balance
            
            # Validation checks
            is_valid = self._validate_position(final_position_size, actual_risk_pct, symbol)
            
            result = {
                "symbol": symbol,
                "side": side,
                "entry_price": entry_price,
                "stop_loss_price": stop_loss_price,
                "position_size_lots": final_position_size,
                "position_size_units": final_position_size * self.contract_sizes["FOREX"],
                "stop_loss_pips": stop_loss_pips,
                "pip_value": pip_value,
                "risk_amount": actual_risk,
                "risk_percentage": actual_risk_pct * 100,
                "max_allowed_lots": max_lots,
                "is_valid": is_valid,
                "calculated_at": datetime.utcnow()
            }
            
            self.logger.info(f"Position sizing for {symbol}: {final_position_size:.3f} lots, "
                           f"Risk: ${actual_risk:.2f} ({actual_risk_pct*100:.2f}%)")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Position sizing error: {e}")
            return self._get_error_result(symbol, str(e))
    
    def _get_pip_value(self, symbol: str, price: float) -> float:
        """
        Calculate pip value for different currency pairs
        
        Args:
            symbol: Currency pair
            price: Current price
            
        Returns:
            Pip value in USD per standard lot
        """
        # For USD-based pairs
        if symbol.endswith("USD"):
            if "JPY" in symbol:
                return 1000.0 / price  # For USDJPY
            else:
                return 10.0  # For EURUSD, GBPUSD, etc.
                
        # For JPY pairs (not USD quoted)
        elif symbol.endswith("JPY"):
            return 10.0  # Standard for EURJPY, GBPJPY, etc.
            
        # For cross pairs
        elif symbol == "EURGBP":
            return 10.0  # Approximate for GBP-based account
            
        # Default fallback
        return 10.0
    
    def _get_pip_size(self, symbol: str) -> float:
        """
        Get pip size for different currency pairs
        
        Args:
            symbol: Currency pair
            
        Returns:
            Pip size (0.0001 for most pairs, 0.01 for JPY pairs)
        """
        if "JPY" in symbol:
            return 0.01  # Japanese Yen pairs
        else:
            return 0.0001  # Most major pairs
    
    def _validate_position(self, position_size: float, risk_pct: float, symbol: str) -> bool:
        """
        Validate position against FTMO rules
        
        Args:
            position_size: Position size in lots
            risk_pct: Risk percentage
            symbol: Currency pair
            
        Returns:
            True if position is valid, False otherwise
        """
        # Check maximum position size
        if position_size > self.max_position_limits.get(symbol, 1.0):
            self.logger.warning(f"Position size {position_size} exceeds limit for {symbol}")
            return False
            
        # Check maximum risk per trade (2% for FTMO)
        if risk_pct > self.max_risk_per_trade:
            self.logger.warning(f"Risk {risk_pct*100:.2f}% exceeds maximum {self.max_risk_per_trade*100}%")
            return False
            
        # Check minimum position size
        if position_size < 0.01:  # Minimum 0.01 lots
            self.logger.warning(f"Position size {position_size} below minimum 0.01 lots")
            return False
            
        return True
    
    def _get_error_result(self, symbol: str, error_msg: str) -> Dict:
        """
        Return error result structure
        """
        return {
            "symbol": symbol,
            "position_size_lots": 0.0,
            "is_valid": False,
            "error": error_msg,
            "calculated_at": datetime.utcnow()
        }
    
    def update_account_balance(self, new_balance: float):
        """
        Update account balance for position sizing calculations
        
        Args:
            new_balance: New account balance
        """
        self.account_balance = new_balance
        self.logger.info(f"Account balance updated to ${new_balance:.2f}")
    
    def get_max_position_info(self, symbol: str) -> Dict:
        """
        Get maximum position information for a symbol
        
        Args:
            symbol: Currency pair
            
        Returns:
            Dict with maximum position details
        """
        max_lots = self.max_position_limits.get(symbol, 1.0)
        max_risk = self.account_balance * self.max_risk_per_trade
        
        return {
            "symbol": symbol,
            "max_lots": max_lots,
            "max_risk_amount": max_risk,
            "max_risk_percentage": self.max_risk_per_trade * 100,
            "account_balance": self.account_balance
        }

# Test the position sizer
if __name__ == "__main__":
    # Initialize with $100,000 account
    sizer = FTMOPositionSizer(account_balance=100000.0)
    
    # Test EURUSD position sizing
    result = sizer.calculate_position_size(
        entry_price=1.0850,
        stop_loss_price=1.0800,  # 50 pip stop loss
        symbol="EURUSD",
        side="BUY"
    )
    
    print("FTMO Position Sizing Test Result:")
    print(f"Symbol: {result['symbol']}")
    print(f"Position Size: {result['position_size_lots']:.3f} lots")
    print(f"Risk Amount: ${result['risk_amount']:.2f}")
    print(f"Risk Percentage: {result['risk_percentage']:.2f}%")
    print(f"Stop Loss: {result['stop_loss_pips']:.1f} pips")
    print(f"Valid: {result['is_valid']}")