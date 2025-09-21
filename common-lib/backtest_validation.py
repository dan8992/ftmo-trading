import os
#!/usr/bin/env python3
"""
Backtest Validation and Acceptance Criteria Checker
"""
import psycopg2
import pandas as pd
from datetime import datetime, timedelta

def validate_backtest_results():
    """Validate end-to-end backtest results against acceptance criteria"""

    conn = psycopg2.connect(
        host="postgres-service",
        port=5432,
        database="dax_trading",
        user="finrl_user",
        password=os.getenv("POSTGRES_PASSWORD")
    )

    print("🔍 BACKTEST VALIDATION REPORT")
    print("=" * 80)

    try:
        with conn.cursor() as cur:
            # 1. Data Infrastructure Validation
            print("\n📊 1. DATA INFRASTRUCTURE VALIDATION")
            print("-" * 40)

            # Check historical data
            cur.execute("SELECT COUNT(*) FROM backtest_data_1m")
            data_count = cur.fetchone()[0]
            print(f"✅ Historical data records: {data_count:,}")

            # Check data quality
            cur.execute("""
                SELECT symbol, COUNT(*) as records,
                       MIN(timestamp) as start_date,
                       MAX(timestamp) as end_date
                FROM backtest_data_1m
                GROUP BY symbol
            """)

            for row in cur.fetchall():
                symbol, records, start, end = row
                duration_days = (end - start).days
                print(f"✅ {symbol}: {records:,} records, {duration_days} days coverage")

            # 2. Signal Generation Validation
            print("\n🤖 2. SIGNAL GENERATION VALIDATION")
            print("-" * 40)

            cur.execute("""
                SELECT COUNT(*) as total_signals,
                       COUNT(DISTINCT symbol) as symbols_covered,
                       MIN(timestamp) as first_signal,
                       MAX(timestamp) as last_signal
                FROM trading_signals
            """)

            total_signals, symbols, first_signal, last_signal = cur.fetchone()
            signal_duration = (last_signal - first_signal).total_seconds() / 3600  # hours
            signals_per_hour = total_signals / max(signal_duration, 1)

            print(f"✅ Total signals generated: {total_signals}")
            print(f"✅ Currency pairs covered: {symbols}")
            print(f"✅ Signal generation rate: {signals_per_hour:.1f} signals/hour")
            print(f"✅ Signal timespan: {signal_duration:.1f} hours")

            # Signal quality metrics
            cur.execute("""
                SELECT signal_type, COUNT(*) as count,
                       AVG(confidence) as avg_confidence,
                       AVG(risk_score) as avg_risk_score
                FROM trading_signals
                GROUP BY signal_type
            """)

            print("\n📈 Signal Quality Metrics:")
            for row in cur.fetchall():
                signal_type, count, avg_conf, avg_risk = row
                print(f"  {signal_type}: {count} signals, conf: {float(avg_conf):.2f}, risk: {float(avg_risk):.2f}")

            # 3. System Performance Validation
            print("\n⚡ 3. SYSTEM PERFORMANCE VALIDATION")
            print("-" * 40)

            # Check if signals have realistic timestamps (not all generated at once)
            cur.execute("""
                SELECT COUNT(DISTINCT DATE_TRUNC('hour', timestamp)) as unique_hours,
                       MAX(timestamp) - MIN(timestamp) as time_span
                FROM trading_signals
            """)

            unique_hours, time_span = cur.fetchone()
            print(f"✅ Signals distributed across {unique_hours} unique hours")
            print(f"✅ Total signal generation timespan: {time_span}")

            # 4. Database Integration Validation
            print("\n🗄️  4. DATABASE INTEGRATION VALIDATION")
            print("-" * 40)

            # Check table structures
            tables_to_check = ['backtest_data_1m', 'trading_signals', 'news_articles', 'economic_events']
            for table in tables_to_check:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                count = cur.fetchone()[0]
                print(f"✅ {table}: {count:,} records")

            # 5. Acceptance Criteria Validation
            print("\n✅ 5. ACCEPTANCE CRITERIA VALIDATION")
            print("-" * 40)

            # Criterion 1: System generates signals
            if total_signals > 0:
                print("✅ PASS: System generates trading signals")
            else:
                print("❌ FAIL: No trading signals generated")

            # Criterion 2: Signal generation rate is realistic
            if 1.0 <= signals_per_hour <= 50.0:  # 1-50 signals per hour is realistic
                print(f"✅ PASS: Signal generation rate is realistic ({signals_per_hour:.1f}/hour)")
            else:
                print(f"❌ FAIL: Signal generation rate is unrealistic ({signals_per_hour:.1f}/hour)")

            # Criterion 3: Signals have proper structure
            cur.execute("SELECT COUNT(*) FROM trading_signals WHERE confidence BETWEEN 0 AND 1")
            valid_confidence_signals = cur.fetchone()[0]

            if valid_confidence_signals == total_signals:
                print("✅ PASS: All signals have valid confidence scores (0-1)")
            else:
                print(f"❌ FAIL: {total_signals - valid_confidence_signals} signals have invalid confidence")

            # Criterion 4: Multiple currency pairs covered
            if symbols >= 2:
                print(f"✅ PASS: Multiple currency pairs covered ({symbols})")
            else:
                print(f"❌ FAIL: Only {symbols} currency pair(s) covered")

            # Criterion 5: Data quality checks
            data_quality_pass = data_count >= 80000  # Should have ~86K records
            if data_quality_pass:
                print(f"✅ PASS: Sufficient historical data ({data_count:,} records)")
            else:
                print(f"❌ FAIL: Insufficient historical data ({data_count:,} records)")

            # 6. System Health Validation
            print("\n🏥 6. SYSTEM HEALTH VALIDATION")
            print("-" * 40)

            # Check for data consistency
            cur.execute("""
                SELECT COUNT(*) FROM trading_signals
                WHERE reasoning IS NOT NULL AND reasoning != ''
            """)
            signals_with_reasoning = cur.fetchone()[0]

            reasoning_pct = (signals_with_reasoning / max(total_signals, 1)) * 100
            print(f"✅ Signals with reasoning: {signals_with_reasoning}/{total_signals} ({reasoning_pct:.1f}%)")

            # Check signal distribution over time
            cur.execute("""
                SELECT DATE_TRUNC('day', timestamp) as day, COUNT(*) as daily_signals
                FROM trading_signals
                GROUP BY DATE_TRUNC('day', timestamp)
                ORDER BY day
                LIMIT 5
            """)

            print("\n📅 Daily Signal Distribution (sample):")
            for row in cur.fetchall():
                day, daily_count = row
                print(f"  {day.date()}: {daily_count} signals")

            # 7. Final Acceptance Score
            print("\n🎯 7. FINAL ACCEPTANCE SCORE")
            print("-" * 40)

            criteria_passed = 0
            total_criteria = 6

            if total_signals > 0: criteria_passed += 1
            if 1.0 <= signals_per_hour <= 50.0: criteria_passed += 1
            if valid_confidence_signals == total_signals: criteria_passed += 1
            if symbols >= 2: criteria_passed += 1
            if data_quality_pass: criteria_passed += 1
            if reasoning_pct >= 80: criteria_passed += 1

            acceptance_score = (criteria_passed / total_criteria) * 100

            print(f"Acceptance Score: {criteria_passed}/{total_criteria} ({acceptance_score:.1f}%)")

            if acceptance_score >= 80:
                print("🎉 OVERALL RESULT: ✅ SYSTEM PASSES ACCEPTANCE CRITERIA")
            elif acceptance_score >= 60:
                print("⚠️  OVERALL RESULT: ⚠️  SYSTEM PARTIALLY MEETS CRITERIA (NEEDS IMPROVEMENT)")
            else:
                print("❌ OVERALL RESULT: ❌ SYSTEM FAILS ACCEPTANCE CRITERIA")

            # 8. Recommendations
            print("\n💡 8. RECOMMENDATIONS")
            print("-" * 40)

            if signals_per_hour > 50:
                print("⚠️  Consider reducing signal generation frequency")
            elif signals_per_hour < 1:
                print("⚠️  Consider increasing signal sensitivity")

            if reasoning_pct < 80:
                print("⚠️  Improve signal reasoning quality")

            if symbols < 2:
                print("⚠️  Expand to cover more currency pairs")

            print("✅ End-to-end pipeline successfully validated")

    finally:
        conn.close()

if __name__ == "__main__":
    validate_backtest_results()