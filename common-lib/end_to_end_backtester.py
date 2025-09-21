import os
#!/usr/bin/env python3
"""
End-to-End FTMO Backtesting Engine
Tests the complete pipeline: Historical Data → LLM Signals → Simulated Trading → FTMO Compliance
"""
import asyncio
import psycopg2
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BacktestConfig:
    start_date: datetime
    end_date: datetime
    initial_balance: float = 100000.0  # $100K FTMO account
    max_daily_loss_pct: float = 0.05   # 5% daily loss limit
    max_total_loss_pct: float = 0.10   # 10% total loss limit
    max_position_size_pct: float = 0.02  # 2% risk per trade
    symbols: List[str] = None

    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ['EURUSD', 'GBPUSD']

@dataclass
class Trade:
    id: int
    timestamp: datetime
    symbol: str
    direction: str  # BUY/SELL
    entry_price: float
    size: float
    exit_price: float = None
    exit_timestamp: datetime = None
    pnl: float = 0.0
    status: str = 'OPEN'  # OPEN/CLOSED

class FTMOBacktester:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.db_config = {
            'host': 'postgres-service',
            'port': 5432,
            'database': 'dax_trading',
            'user': 'finrl_user',
            'password': os.getenv('POSTGRES_PASSWORD')
        }
        
        # Trading state
        self.current_balance = config.initial_balance
        self.peak_balance = config.initial_balance
        self.daily_start_balance = config.initial_balance
        self.open_trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.compliance_violations = []
        
        # Performance tracking
        self.signals_processed = 0
        self.trades_executed = 0
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        
    def connect_db(self):
        """Get database connection"""
        return psycopg2.connect(**self.db_config)

    def get_historical_price(self, symbol: str, timestamp: datetime) -> Optional[Dict]:
        """Get OHLC price at specific timestamp"""
        conn = self.connect_db()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT open, high, low, close, volume
                    FROM backtest_data_1m 
                    WHERE symbol = %s 
                        AND timestamp <= %s 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """, (symbol, timestamp))
                
                row = cur.fetchone()
                if row:
                    return {
                        'open': float(row[0]),
                        'high': float(row[1]),
                        'low': float(row[2]),
                        'close': float(row[3]),
                        'volume': int(row[4])
                    }
                return None
        finally:
            conn.close()

    def simulate_llm_signal_generation(self, timestamp: datetime, symbol: str) -> Optional[Dict]:
        """Simulate LLM signal generation based on market conditions"""
        price_data = self.get_historical_price(symbol, timestamp)
        if not price_data:
            return None
        
        # Get recent price movement for signal generation
        conn = self.connect_db()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT close 
                    FROM backtest_data_1m 
                    WHERE symbol = %s 
                        AND timestamp BETWEEN %s AND %s
                    ORDER BY timestamp ASC
                """, (symbol, timestamp - timedelta(hours=1), timestamp))
                
                recent_prices = [float(row[0]) for row in cur.fetchall()]
                
        finally:
            conn.close()
        
        if len(recent_prices) < 10:
            return None
        
        # Simple momentum-based signal generation
        current_price = recent_prices[-1]
        avg_price = np.mean(recent_prices)
        price_change = (current_price - avg_price) / avg_price
        volatility = np.std(recent_prices) / avg_price
        
        # Generate signal based on price momentum
        if price_change > 0.0005:  # 5 pips up
            signal_type = 'BUY'
            confidence = float(min(0.7 + abs(price_change) * 100, 0.95))
        elif price_change < -0.0005:  # 5 pips down
            signal_type = 'SELL'
            confidence = float(min(0.7 + abs(price_change) * 100, 0.95))
        else:
            signal_type = 'HOLD'
            confidence = 0.3
        
        # Risk assessment
        risk_score = float(volatility * 100)  # Higher volatility = higher risk
        
        if signal_type != 'HOLD' and confidence > 0.6:
            return {
                'timestamp': timestamp,
                'symbol': symbol,
                'signal_type': signal_type,
                'confidence': confidence,
                'risk_score': risk_score,
                'reasoning': f'Price momentum: {price_change:.5f}, volatility: {volatility:.5f}',
                'entry_price': float(current_price)
            }
        
        return None

    def calculate_position_size(self, signal: Dict) -> float:
        """Calculate position size based on FTMO risk management"""
        confidence = signal['confidence']
        risk_score = signal['risk_score']
        
        # Base position size as percentage of balance
        base_size_pct = self.config.max_position_size_pct
        
        # Adjust based on confidence (higher confidence = larger position)
        confidence_multiplier = confidence
        
        # Adjust based on risk (higher risk = smaller position)
        risk_multiplier = max(0.5, 1.0 - risk_score)
        
        # Calculate position size
        position_size_pct = base_size_pct * confidence_multiplier * risk_multiplier
        position_size = self.current_balance * position_size_pct
        
        # Ensure we don't exceed maximum position size
        max_position = self.current_balance * self.config.max_position_size_pct
        position_size = min(position_size, max_position)
        
        return position_size

    def check_ftmo_compliance(self, potential_loss: float = 0) -> Tuple[bool, List[str]]:
        """Check FTMO compliance rules"""
        violations = []
        
        # Check daily loss limit
        daily_loss = self.daily_start_balance - (self.current_balance + potential_loss)
        daily_loss_pct = daily_loss / self.config.initial_balance
        
        if daily_loss_pct > self.config.max_daily_loss_pct:
            violations.append(f"Daily loss limit exceeded: {daily_loss_pct:.2%} > {self.config.max_daily_loss_pct:.2%}")
        
        # Check total loss limit
        total_loss = self.config.initial_balance - (self.current_balance + potential_loss)
        total_loss_pct = total_loss / self.config.initial_balance
        
        if total_loss_pct > self.config.max_total_loss_pct:
            violations.append(f"Total loss limit exceeded: {total_loss_pct:.2%} > {self.config.max_total_loss_pct:.2%}")
        
        return len(violations) == 0, violations

    def execute_trade(self, signal: Dict) -> Optional[Trade]:
        """Execute trade based on signal"""
        if signal['signal_type'] == 'HOLD':
            return None
        
        position_size = self.calculate_position_size(signal)
        
        # Check if we can afford this trade
        if position_size < 1000:  # Minimum $1000 position
            return None
        
        # Check FTMO compliance before trade
        max_loss = position_size * 0.02  # Assume 2% max loss per trade
        compliant, violations = self.check_ftmo_compliance(potential_loss=-max_loss)
        
        if not compliant:
            logger.warning(f"Trade blocked due to FTMO violations: {violations}")
            self.compliance_violations.extend(violations)
            return None
        
        # Create trade
        trade = Trade(
            id=len(self.closed_trades) + len(self.open_trades) + 1,
            timestamp=signal['timestamp'],
            symbol=signal['symbol'],
            direction=signal['signal_type'],
            entry_price=signal['entry_price'],
            size=position_size
        )
        
        self.open_trades.append(trade)
        self.trades_executed += 1
        
        logger.info(f"Executed trade {trade.id}: {trade.direction} {trade.symbol} @ {trade.entry_price} (size: ${trade.size:,.0f})")
        
        return trade

    def update_open_trades(self, timestamp: datetime):
        """Update P&L for open trades and close if needed"""
        trades_to_close = []
        
        for trade in self.open_trades:
            current_price_data = self.get_historical_price(trade.symbol, timestamp)
            if not current_price_data:
                continue
            
            current_price = current_price_data['close']
            
            # Calculate unrealized P&L
            if trade.direction == 'BUY':
                price_change = current_price - trade.entry_price
            else:  # SELL
                price_change = trade.entry_price - current_price
            
            # Calculate P&L (simplified - no spread/commission for backtest)
            pip_value = 10 if 'JPY' in trade.symbol else 100000  # $10 per pip for majors
            pnl = price_change * pip_value * (trade.size / 100000)
            
            # Close trade if we hit stop loss (2% of position) or take profit (4% of position)
            stop_loss = trade.size * -0.02
            take_profit = trade.size * 0.04
            
            # Close after maximum 4 hours or if stop/target hit
            time_in_trade = timestamp - trade.timestamp
            
            if pnl <= stop_loss or pnl >= take_profit or time_in_trade >= timedelta(hours=4):
                trade.exit_price = current_price
                trade.exit_timestamp = timestamp
                trade.pnl = pnl
                trade.status = 'CLOSED'
                trades_to_close.append(trade)
                
                # Update balance
                self.current_balance += pnl
                self.total_pnl += pnl
                self.daily_pnl += pnl
                
                logger.info(f"Closed trade {trade.id}: P&L = ${pnl:,.2f}, Balance = ${self.current_balance:,.2f}")
        
        # Move closed trades
        for trade in trades_to_close:
            self.open_trades.remove(trade)
            self.closed_trades.append(trade)

    def log_compliance_status(self, timestamp: datetime):
        """Log current compliance status to database"""
        daily_loss_pct = (self.daily_start_balance - self.current_balance) / self.config.initial_balance
        total_loss_pct = (self.config.initial_balance - self.current_balance) / self.config.initial_balance
        
        compliant, violations = self.check_ftmo_compliance()
        
        conn = self.connect_db()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO ftmo_compliance_log 
                    (timestamp, account_balance, daily_pnl, max_daily_loss_limit, total_pnl, max_total_loss_limit, is_compliant, violations)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    timestamp,
                    self.current_balance,
                    -daily_loss_pct * self.config.initial_balance,
                    self.config.max_daily_loss_pct * self.config.initial_balance,
                    self.total_pnl,
                    self.config.max_total_loss_pct * self.config.initial_balance,
                    compliant,
                    violations
                ))
                conn.commit()
        finally:
            conn.close()

    def store_backtest_signal(self, signal: Dict):
        """Store generated signal in database"""
        if not signal:
            return
            
        conn = self.connect_db()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO trading_signals 
                    (timestamp, symbol, signal_type, confidence, reasoning, technical_indicators, risk_score)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    signal['timestamp'],
                    signal['symbol'],
                    signal['signal_type'],
                    signal['confidence'],
                    signal['reasoning'],
                    json.dumps({'entry_price': signal['entry_price'], 'backtest_mode': True}),
                    signal['risk_score']
                ))
                conn.commit()
        finally:
            conn.close()

    async def run_backtest(self):
        """Run complete end-to-end backtest"""
        logger.info(f"Starting backtest from {self.config.start_date} to {self.config.end_date}")
        logger.info(f"Initial balance: ${self.config.initial_balance:,.2f}")
        
        current_time = self.config.start_date
        last_daily_reset = current_time.date()
        
        while current_time <= self.config.end_date:
            # Reset daily tracking if new day
            if current_time.date() != last_daily_reset:
                self.daily_start_balance = self.current_balance
                self.daily_pnl = 0.0
                last_daily_reset = current_time.date()
                logger.info(f"New trading day: {current_time.date()}, Starting balance: ${self.current_balance:,.2f}")
            
            # Process each symbol
            for symbol in self.config.symbols:
                # Generate LLM signal
                signal = self.simulate_llm_signal_generation(current_time, symbol)
                
                if signal:
                    self.signals_processed += 1
                    self.store_backtest_signal(signal)
                    
                    # Execute trade if signal is actionable
                    trade = self.execute_trade(signal)
                
                # Update open trades
                self.update_open_trades(current_time)
            
            # Log compliance every hour
            if current_time.minute == 0:
                self.log_compliance_status(current_time)
            
            # Check if we've hit FTMO limits
            compliant, violations = self.check_ftmo_compliance()
            if not compliant:
                logger.error(f"FTMO compliance violation at {current_time}: {violations}")
                self.compliance_violations.extend(violations)
                break
            
            # Move to next minute
            current_time += timedelta(minutes=5)  # Process every 5 minutes for speed
        
        # Close any remaining open trades
        for trade in self.open_trades:
            final_price_data = self.get_historical_price(trade.symbol, self.config.end_date)
            if final_price_data:
                trade.exit_price = final_price_data['close']
                trade.exit_timestamp = self.config.end_date
                
                if trade.direction == 'BUY':
                    price_change = trade.exit_price - trade.entry_price
                else:
                    price_change = trade.entry_price - trade.exit_price
                
                pip_value = 10 if 'JPY' in trade.symbol else 100000
                trade.pnl = price_change * pip_value * (trade.size / 100000)
                trade.status = 'CLOSED'
                
                self.current_balance += trade.pnl
                self.total_pnl += trade.pnl
                self.closed_trades.append(trade)
        
        self.open_trades = []
        
        # Final compliance check
        self.log_compliance_status(self.config.end_date)
        
        logger.info("Backtest completed!")
        self.print_results()

    def print_results(self):
        """Print backtest results"""
        print("\n" + "="*80)
        print("📊 BACKTEST RESULTS")
        print("="*80)
        
        # Performance metrics
        total_return = (self.current_balance - self.config.initial_balance) / self.config.initial_balance
        max_drawdown = (self.peak_balance - min(self.peak_balance, self.current_balance)) / self.peak_balance
        
        print(f"Initial Balance:     ${self.config.initial_balance:,.2f}")
        print(f"Final Balance:       ${self.current_balance:,.2f}")
        print(f"Total P&L:          ${self.total_pnl:,.2f}")
        print(f"Total Return:        {total_return:.2%}")
        print(f"Max Drawdown:        {max_drawdown:.2%}")
        print(f"Signals Processed:   {self.signals_processed}")
        print(f"Trades Executed:     {self.trades_executed}")
        print(f"Trades Closed:       {len(self.closed_trades)}")
        print(f"Compliance Violations: {len(self.compliance_violations)}")
        
        if self.closed_trades:
            winning_trades = [t for t in self.closed_trades if t.pnl > 0]
            losing_trades = [t for t in self.closed_trades if t.pnl <= 0]
            
            win_rate = len(winning_trades) / len(self.closed_trades) if self.closed_trades else 0
            avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
            
            print(f"Win Rate:            {win_rate:.1%}")
            print(f"Average Win:         ${avg_win:.2f}")
            print(f"Average Loss:        ${avg_loss:.2f}")
        
        # FTMO Compliance Check
        print(f"\n🛡️  FTMO COMPLIANCE:")
        daily_loss = (self.daily_start_balance - self.current_balance) / self.config.initial_balance
        total_loss = (self.config.initial_balance - self.current_balance) / self.config.initial_balance
        
        print(f"Daily Loss:          {daily_loss:.2%} (Limit: {self.config.max_daily_loss_pct:.2%})")
        print(f"Total Loss:          {total_loss:.2%} (Limit: {self.config.max_total_loss_pct:.2%})")
        print(f"Status:              {'✅ PASSED' if len(self.compliance_violations) == 0 else '❌ FAILED'}")
        
        if self.compliance_violations:
            print(f"\n❌ Violations:")
            for violation in self.compliance_violations:
                print(f"  - {violation}")

if __name__ == "__main__":
    # Configure backtest
    config = BacktestConfig(
        start_date=datetime.now() - timedelta(days=7),  # Last 7 days
        end_date=datetime.now() - timedelta(days=1),    # Until yesterday
        initial_balance=100000.0,
        symbols=['EURUSD', 'GBPUSD']
    )
    
    # Run backtest
    backtester = FTMOBacktester(config)
    asyncio.run(backtester.run_backtest())