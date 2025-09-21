import os
#!/usr/bin/env python3
"""
FTMO Market Event and Time Filtering
Implements news event filtering and market hours validation
"""
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, time, timedelta
from enum import Enum
import calendar

class MarketSession(Enum):
    SYDNEY = "SYDNEY"
    TOKYO = "TOKYO"
    LONDON = "LONDON"
    NEW_YORK = "NEW_YORK"
    CLOSED = "CLOSED"

class NewsImpact(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"

class TradingRecommendation(Enum):
    PROCEED = "PROCEED"
    REDUCE_SIZE = "REDUCE_SIZE"
    AVOID_TRADING = "AVOID_TRADING"
    NO_TRADING = "NO_TRADING"

class FTMOMarketFilter:
    """
    Filter trading opportunities based on market conditions and news events
    """
    
    def __init__(self):
        """
        Initialize market filter with default settings
        """
        # Market session times (UTC)
        self.market_sessions = {
            MarketSession.SYDNEY: (time(21, 0), time(6, 0)),      # 9 PM - 6 AM UTC
            MarketSession.TOKYO: (time(23, 0), time(8, 0)),       # 11 PM - 8 AM UTC
            MarketSession.LONDON: (time(8, 0), time(17, 0)),      # 8 AM - 5 PM UTC
            MarketSession.NEW_YORK: (time(13, 0), time(22, 0)),   # 1 PM - 10 PM UTC
        }
        
        # High-impact news events (simplified calendar)
        self.regular_news_events = {
            # US Events (UTC times)
            "NFP": {"day": 4, "time": time(12, 30), "impact": NewsImpact.EXTREME},  # First Friday
            "FOMC": {"day": 2, "time": time(18, 0), "impact": NewsImpact.EXTREME}, # Wednesday
            "CPI": {"time": time(12, 30), "impact": NewsImpact.HIGH},
            "PPI": {"time": time(12, 30), "impact": NewsImpact.MEDIUM},
            "GDP": {"time": time(12, 30), "impact": NewsImpact.HIGH},
            "RETAIL_SALES": {"time": time(12, 30), "impact": NewsImpact.MEDIUM},
            "UNEMPLOYMENT": {"time": time(12, 30), "impact": NewsImpact.MEDIUM},
            
            # EUR Events
            "ECB_RATE": {"day": 3, "time": time(11, 45), "impact": NewsImpact.EXTREME}, # Thursday
            "ECB_PRESS": {"day": 3, "time": time(12, 30), "impact": NewsImpact.HIGH},
            "EUR_CPI": {"time": time(9, 0), "impact": NewsImpact.HIGH},
            "EUR_GDP": {"time": time(9, 0), "impact": NewsImpact.MEDIUM},
            
            # GBP Events
            "BOE_RATE": {"day": 3, "time": time(11, 0), "impact": NewsImpact.EXTREME}, # Thursday
            "GBP_CPI": {"time": time(6, 0), "impact": NewsImpact.HIGH},
            "GBP_GDP": {"time": time(6, 0), "impact": NewsImpact.MEDIUM},
            
            # JPY Events
            "BOJ_RATE": {"time": time(3, 0), "impact": NewsImpact.HIGH},
            "JPY_CPI": {"time": time(23, 30), "impact": NewsImpact.MEDIUM},
        }
        
        # Market holiday calendar (major holidays)
        self.market_holidays = {
            "NEW_YEAR": (1, 1),
            "CHRISTMAS": (12, 25),
            "THANKSGIVING_US": (11, 22),  # Approximate - 4th Thursday
        }
        
        # Variable holidays (handled separately)
        self.variable_holidays = ["GOOD_FRIDAY", "EASTER_MONDAY"]
        
        # Volatility periods to avoid
        self.avoid_periods = {
            "MARKET_OPEN": 30,      # First 30 minutes of major sessions
            "MARKET_CLOSE": 30,     # Last 30 minutes of major sessions
            "ROLLOVER": 60,         # 1 hour around daily rollover (5 PM EST)
            "NEWS_BUFFER": 30,      # 30 minutes before/after major news
        }
        
        # Low liquidity periods
        self.low_liquidity_periods = [
            (time(22, 0), time(23, 0)),   # Between NY close and Sydney open
            (time(6, 0), time(8, 0)),     # Between Sydney close and London open
        ]
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def is_trading_allowed(self, timestamp: datetime, symbol: str,
                          risk_level: str = "conservative") -> Dict:
        """
        Comprehensive trading permission check
        
        Args:
            timestamp: Time to check (UTC)
            symbol: Currency pair
            risk_level: "conservative", "moderate", or "aggressive"
            
        Returns:
            Dict with permission status and reasons
        """
        results = {
            "timestamp": timestamp,
            "symbol": symbol,
            "risk_level": risk_level,
            "allowed": True,
            "recommendation": TradingRecommendation.PROCEED,
            "reasons": [],
            "warnings": [],
            "market_conditions": {}
        }
        
        # Check 1: Weekend trading
        weekend_check = self._check_weekend(timestamp)
        results["market_conditions"]["weekend"] = weekend_check
        if not weekend_check["trading_allowed"]:
            results["allowed"] = False
            results["recommendation"] = TradingRecommendation.NO_TRADING
            results["reasons"].append("WEEKEND_MARKET_CLOSED")
        
        # Check 2: Market holidays
        holiday_check = self._check_holidays(timestamp)
        results["market_conditions"]["holiday"] = holiday_check
        if holiday_check["is_holiday"]:
            if risk_level == "conservative":
                results["allowed"] = False
                results["recommendation"] = TradingRecommendation.NO_TRADING
                results["reasons"].append(f"HOLIDAY_{holiday_check['holiday_name']}")
            else:
                results["warnings"].append(f"TRADING_ON_HOLIDAY_{holiday_check['holiday_name']}")
        
        # Check 3: Market sessions and liquidity
        session_check = self._check_market_sessions(timestamp, symbol)
        results["market_conditions"]["session"] = session_check
        if session_check["liquidity"] == "VERY_LOW":
            if risk_level == "conservative":
                results["allowed"] = False
                results["recommendation"] = TradingRecommendation.NO_TRADING
                results["reasons"].append("VERY_LOW_LIQUIDITY")
            else:
                results["recommendation"] = TradingRecommendation.REDUCE_SIZE
                results["warnings"].append("LOW_LIQUIDITY_PERIOD")
        
        # Check 4: News events
        news_check = self._check_news_events(timestamp, symbol, risk_level)
        results["market_conditions"]["news"] = news_check
        if news_check["high_impact_soon"]:
            if risk_level == "conservative":
                results["allowed"] = False
                results["recommendation"] = TradingRecommendation.NO_TRADING
                results["reasons"].append(f"HIGH_IMPACT_NEWS_{news_check['next_event']['name']}")
            else:
                results["recommendation"] = TradingRecommendation.REDUCE_SIZE
                results["warnings"].append(f"NEWS_EVENT_APPROACHING_{news_check['next_event']['name']}")
        
        # Check 5: Volatility periods
        volatility_check = self._check_volatility_periods(timestamp)
        results["market_conditions"]["volatility"] = volatility_check
        if volatility_check["high_volatility_period"]:
            if risk_level == "conservative":
                results["recommendation"] = TradingRecommendation.REDUCE_SIZE
                results["warnings"].append(f"HIGH_VOLATILITY_{volatility_check['period_type']}")
        
        # Final recommendation logic
        if results["allowed"] and len(results["warnings"]) > 2:
            results["recommendation"] = TradingRecommendation.AVOID_TRADING
            results["reasons"].append("MULTIPLE_RISK_FACTORS")
        
        self.logger.info(f"Trading check for {symbol} at {timestamp}: "
                        f"{'ALLOWED' if results['allowed'] else 'BLOCKED'} "
                        f"({results['recommendation'].value})")
        
        return results
    
    def _check_weekend(self, timestamp: datetime) -> Dict:
        """
        Check if timestamp falls on weekend
        """
        weekday = timestamp.weekday()  # 0=Monday, 6=Sunday
        
        # Saturday and Sunday
        if weekday >= 5:
            return {
                "trading_allowed": False,
                "is_weekend": True,
                "weekday": weekday,
                "reason": "WEEKEND_MARKET_CLOSED"
            }
        
        # Friday after 22:00 UTC (start of weekend)
        if weekday == 4 and timestamp.time() >= time(22, 0):
            return {
                "trading_allowed": False,
                "is_weekend": True,
                "weekday": weekday,
                "reason": "WEEKEND_STARTING"
            }
        
        # Sunday before 22:00 UTC (weekend still active)
        if weekday == 6 and timestamp.time() < time(22, 0):
            return {
                "trading_allowed": False,
                "is_weekend": True,
                "weekday": weekday,
                "reason": "WEEKEND_ENDING"
            }
        
        return {
            "trading_allowed": True,
            "is_weekend": False,
            "weekday": weekday
        }
    
    def _check_holidays(self, timestamp: datetime) -> Dict:
        """
        Check if timestamp falls on a market holiday
        """
        month = timestamp.month
        day = timestamp.day
        
        # Check fixed holidays
        for holiday_name, (holiday_month, holiday_day) in self.market_holidays.items():
            if month == holiday_month and day == holiday_day:
                return {
                    "is_holiday": True,
                    "holiday_name": holiday_name,
                    "impact": "MARKET_CLOSED"
                }
        
        # Check for Thanksgiving (4th Thursday of November)
        if month == 11:
            # Find 4th Thursday
            first_day = datetime(timestamp.year, 11, 1)
            first_thursday = 3 - first_day.weekday() if first_day.weekday() <= 3 else 10 - first_day.weekday()
            fourth_thursday = first_thursday + 21  # 3 weeks later
            
            if day == fourth_thursday:
                return {
                    "is_holiday": True,
                    "holiday_name": "THANKSGIVING_US",
                    "impact": "EARLY_CLOSE"
                }
        
        return {
            "is_holiday": False,
            "holiday_name": None
        }
    
    def _check_market_sessions(self, timestamp: datetime, symbol: str) -> Dict:
        """
        Check current market session and liquidity
        """
        current_time = timestamp.time()
        active_sessions = []
        
        # Check which sessions are active
        for session, (start_time, end_time) in self.market_sessions.items():
            if start_time <= end_time:  # Same day
                if start_time <= current_time <= end_time:
                    active_sessions.append(session)
            else:  # Crosses midnight
                if current_time >= start_time or current_time <= end_time:
                    active_sessions.append(session)
        
        # Determine liquidity based on active sessions and symbol
        liquidity = self._calculate_liquidity(active_sessions, symbol, current_time)
        
        return {
            "active_sessions": [s.value for s in active_sessions],
            "primary_session": active_sessions[0].value if active_sessions else MarketSession.CLOSED.value,
            "liquidity": liquidity,
            "overlap_sessions": len(active_sessions) > 1
        }
    
    def _calculate_liquidity(self, active_sessions: List[MarketSession], 
                           symbol: str, current_time: time) -> str:
        """
        Calculate liquidity level based on sessions and symbol
        """
        # No active sessions
        if not active_sessions:
            return "VERY_LOW"
        
        # Check for low liquidity periods
        for start_time, end_time in self.low_liquidity_periods:
            if start_time <= current_time <= end_time:
                return "VERY_LOW"
        
        # Multiple sessions overlap = high liquidity
        if len(active_sessions) >= 2:
            return "HIGH"
        
        # Single session liquidity depends on symbol and session
        session = active_sessions[0]
        
        # EUR pairs during London session
        if session == MarketSession.LONDON and symbol.startswith("EUR"):
            return "HIGH"
        
        # USD pairs during New York session
        if session == MarketSession.NEW_YORK and "USD" in symbol:
            return "HIGH"
        
        # JPY pairs during Tokyo session
        if session == MarketSession.TOKYO and symbol.endswith("JPY"):
            return "MEDIUM"
        
        # AUD pairs during Sydney session
        if session == MarketSession.SYDNEY and symbol.startswith("AUD"):
            return "MEDIUM"
        
        # Default for single session
        return "MEDIUM"
    
    def _check_news_events(self, timestamp: datetime, symbol: str, risk_level: str) -> Dict:
        """
        Check for upcoming high-impact news events
        """
        # Buffer times based on risk level
        buffer_minutes = {
            "conservative": 60,  # 1 hour buffer
            "moderate": 30,      # 30 minute buffer
            "aggressive": 15     # 15 minute buffer
        }
        
        buffer = buffer_minutes.get(risk_level, 30)
        check_window = timedelta(minutes=buffer)
        
        upcoming_events = []
        
        # Check regular events
        for event_name, event_config in self.regular_news_events.items():
            # Check if event affects this symbol
            if not self._event_affects_symbol(event_name, symbol):
                continue
            
            # Calculate next occurrence
            next_occurrence = self._calculate_next_event_time(timestamp, event_config)
            
            if next_occurrence:
                time_until = next_occurrence - timestamp
                
                # Check if event is within buffer window
                if timedelta() <= time_until <= check_window:
                    upcoming_events.append({
                        "name": event_name,
                        "time": next_occurrence,
                        "impact": event_config["impact"].value,
                        "minutes_until": int(time_until.total_seconds() / 60),
                        "affects_symbol": True
                    })
        
        # Determine if high impact event is soon
        high_impact_soon = any(
            event["impact"] in ["HIGH", "EXTREME"] 
            for event in upcoming_events
        )
        
        return {
            "upcoming_events": upcoming_events,
            "high_impact_soon": high_impact_soon,
            "next_event": upcoming_events[0] if upcoming_events else None,
            "buffer_minutes": buffer
        }
    
    def _event_affects_symbol(self, event_name: str, symbol: str) -> bool:
        """
        Check if news event affects the given symbol
        """
        # US events affect USD pairs
        if event_name in ["NFP", "FOMC", "CPI", "PPI", "GDP", "RETAIL_SALES", "UNEMPLOYMENT"]:
            return "USD" in symbol
        
        # EUR events affect EUR pairs
        if event_name.startswith("EUR") or event_name.startswith("ECB"):
            return "EUR" in symbol
        
        # GBP events affect GBP pairs
        if event_name.startswith("GBP") or event_name.startswith("BOE"):
            return "GBP" in symbol
        
        # JPY events affect JPY pairs
        if event_name.startswith("JPY") or event_name.startswith("BOJ"):
            return "JPY" in symbol
        
        return False
    
    def _calculate_next_event_time(self, current_time: datetime, event_config: Dict) -> Optional[datetime]:
        """
        Calculate next occurrence of a news event
        """
        event_time = event_config["time"]
        
        # For events with specific day of week
        if "day" in event_config:
            target_weekday = event_config["day"]
            
            # Find next occurrence of target weekday
            days_ahead = target_weekday - current_time.weekday()
            if days_ahead <= 0:  # Target day already happened this week
                days_ahead += 7
            
            next_date = current_time.date() + timedelta(days=days_ahead)
            next_datetime = datetime.combine(next_date, event_time)
            
            # If it's today but time has passed, move to next week
            if next_datetime <= current_time:
                next_datetime += timedelta(days=7)
            
            return next_datetime
        
        # For events that happen today
        today_event = datetime.combine(current_time.date(), event_time)
        if today_event > current_time:
            return today_event
        
        # Event today has passed, check tomorrow
        tomorrow_event = datetime.combine(current_time.date() + timedelta(days=1), event_time)
        return tomorrow_event
    
    def _check_volatility_periods(self, timestamp: datetime) -> Dict:
        """
        Check for periods of expected high volatility
        """
        current_time = timestamp.time()
        
        # Market open periods (high volatility)
        volatility_periods = [
            (time(8, 0), time(8, 30), "LONDON_OPEN"),     # London open
            (time(13, 0), time(13, 30), "NY_OPEN"),       # New York open
            (time(21, 0), time(21, 30), "SYDNEY_OPEN"),   # Sydney open
            (time(16, 30), time(17, 30), "NY_CLOSE"),     # New York close
        ]
        
        for start_time, end_time, period_name in volatility_periods:
            if start_time <= current_time <= end_time:
                return {
                    "high_volatility_period": True,
                    "period_type": period_name,
                    "reason": "MARKET_OPEN_CLOSE"
                }
        
        # Daily rollover period (5 PM EST = 22:00 UTC)
        if time(21, 30) <= current_time <= time(22, 30):
            return {
                "high_volatility_period": True,
                "period_type": "DAILY_ROLLOVER",
                "reason": "SWAP_ROLLOVER"
            }
        
        return {
            "high_volatility_period": False,
            "period_type": None
        }
    
    def get_optimal_trading_windows(self, symbol: str, date: datetime.date = None) -> List[Dict]:
        """
        Get optimal trading windows for a symbol on a given date
        """
        if date is None:
            date = datetime.now().date()
        
        optimal_windows = []
        
        # Define optimal periods based on symbol
        if "EUR" in symbol:
            # Best during London session
            optimal_windows.append({
                "start": time(8, 30),   # After London open volatility
                "end": time(11, 30),    # Before ECB potential events
                "session": "LONDON_MORNING",
                "liquidity": "HIGH",
                "reason": "EUR_PRIME_TIME"
            })
            optimal_windows.append({
                "start": time(13, 30),  # After NY open volatility
                "end": time(16, 0),     # Before NY close
                "session": "LONDON_NY_OVERLAP",
                "liquidity": "VERY_HIGH",
                "reason": "DUAL_SESSION_OVERLAP"
            })
        
        elif "USD" in symbol:
            # Best during NY session and overlaps
            optimal_windows.append({
                "start": time(13, 30),  # After NY open
                "end": time(16, 0),     # During London-NY overlap
                "session": "NY_LONDON_OVERLAP",
                "liquidity": "VERY_HIGH",
                "reason": "USD_PRIME_TIME"
            })
        
        elif "JPY" in symbol:
            # Best during Tokyo session
            optimal_windows.append({
                "start": time(0, 0),    # Tokyo session
                "end": time(3, 0),      # Before volatility
                "session": "TOKYO_EARLY",
                "liquidity": "HIGH",
                "reason": "JPY_PRIME_TIME"
            })
        
        return optimal_windows
    
    def get_market_condition_summary(self, timestamp: datetime) -> Dict:
        """
        Get comprehensive market condition summary
        """
        weekend_check = self._check_weekend(timestamp)
        holiday_check = self._check_holidays(timestamp)
        session_check = self._check_market_sessions(timestamp, "EURUSD")  # Use major pair
        volatility_check = self._check_volatility_periods(timestamp)
        
        return {
            "timestamp": timestamp,
            "weekend": weekend_check,
            "holiday": holiday_check,
            "session": session_check,
            "volatility": volatility_check,
            "overall_conditions": self._assess_overall_conditions(
                weekend_check, holiday_check, session_check, volatility_check
            )
        }
    
    def _assess_overall_conditions(self, weekend_check: Dict, holiday_check: Dict,
                                 session_check: Dict, volatility_check: Dict) -> Dict:
        """
        Assess overall market conditions
        """
        if not weekend_check["trading_allowed"]:
            return {"status": "MARKET_CLOSED", "recommendation": "NO_TRADING"}
        
        if holiday_check["is_holiday"]:
            return {"status": "HOLIDAY", "recommendation": "AVOID_TRADING"}
        
        if session_check["liquidity"] == "VERY_LOW":
            return {"status": "LOW_LIQUIDITY", "recommendation": "AVOID_TRADING"}
        
        if volatility_check["high_volatility_period"]:
            return {"status": "HIGH_VOLATILITY", "recommendation": "REDUCE_SIZE"}
        
        if session_check["overlap_sessions"]:
            return {"status": "OPTIMAL", "recommendation": "PROCEED"}
        
        return {"status": "NORMAL", "recommendation": "PROCEED"}

# Test the market filter
if __name__ == "__main__":
    filter_system = FTMOMarketFilter()
    
    print("FTMO Market Filter Test:")
    print("=" * 50)
    
    # Test 1: Current time
    now = datetime.utcnow()
    result1 = filter_system.is_trading_allowed(now, "EURUSD", "conservative")
    print(f"Current time ({now.strftime('%Y-%m-%d %H:%M UTC')}):")
    print(f"  Trading allowed: {result1['allowed']}")
    print(f"  Recommendation: {result1['recommendation'].value}")
    print(f"  Reasons: {result1['reasons']}")
    print(f"  Warnings: {result1['warnings']}")
    
    # Test 2: Weekend time
    weekend_time = datetime(2025, 9, 21, 15, 0)  # Saturday 3 PM
    result2 = filter_system.is_trading_allowed(weekend_time, "EURUSD", "conservative")
    print(f"\nWeekend time ({weekend_time.strftime('%Y-%m-%d %H:%M UTC')}):")
    print(f"  Trading allowed: {result2['allowed']}")
    print(f"  Recommendation: {result2['recommendation'].value}")
    
    # Test 3: News event time (simulate NFP Friday at 12:25 UTC)
    nfp_time = datetime(2025, 9, 26, 12, 25)  # 5 minutes before NFP
    result3 = filter_system.is_trading_allowed(nfp_time, "EURUSD", "conservative")
    print(f"\nNear NFP time ({nfp_time.strftime('%Y-%m-%d %H:%M UTC')}):")
    print(f"  Trading allowed: {result3['allowed']}")
    print(f"  Recommendation: {result3['recommendation'].value}")
    
    # Test 4: Get optimal trading windows
    optimal_windows = filter_system.get_optimal_trading_windows("EURUSD")
    print(f"\nOptimal EURUSD trading windows:")
    for window in optimal_windows:
        print(f"  {window['start']} - {window['end']} UTC ({window['reason']})")
    
    # Test 5: Market condition summary
    summary = filter_system.get_market_condition_summary(now)
    print(f"\nMarket conditions summary:")
    print(f"  Overall status: {summary['overall_conditions']['status']}")
    print(f"  Recommendation: {summary['overall_conditions']['recommendation']}")
    print(f"  Active sessions: {summary['session']['active_sessions']}")
    print(f"  Liquidity: {summary['session']['liquidity']}")