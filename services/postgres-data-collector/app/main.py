#!/usr/bin/env python3
"""
PostgreSQL-Only DAX Data Collector
Collects market data and writes directly to PostgreSQL
"""
import yfinance as yf
import psycopg2
import time
import logging
from datetime import datetime, timezone
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('PostgreSQLDataCollector')

# DAX symbols
DAX_SYMBOLS = [
    '^GDAXI',  # DAX Index
    'SAP.DE', 'SIE.DE', 'ALV.DE', 'BMW.DE', 'BAS.DE',
    'ADS.DE', 'VOW3.DE', 'MUV2.DE', 'DBK.DE', 'DB1.DE',
    'IFX.DE', 'HEN3.DE', 'MRK.DE', 'FRE.DE', 'CON.DE',
    'DTE.DE', 'RHM.DE', 'ENR.DE', 'AIR.PA'
]

def get_db_connection():
    """Create PostgreSQL connection"""
    return psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'postgres-service'),
        database=os.getenv('POSTGRES_DB', 'finrl_dax'),
        user=os.getenv('POSTGRES_USER', 'finrl_user'),
        password=os.getenv('POSTGRES_PASSWORD')
    )

def collect_and_store_data():
    """Collect market data and store directly in PostgreSQL"""
    logger.info("Starting market data collection...")

    conn = get_db_connection()
    cursor = conn.cursor()

    collected = 0
    for symbol in DAX_SYMBOLS:
        try:
            # Fetch data from Yahoo Finance
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval="1m")

            if data.empty:
                logger.warning(f"No data for {symbol}")
                continue

            # Get latest price
            latest = data.iloc[-1]

            market_data = {
                'symbol': symbol,
                'timestamp': datetime.now(timezone.utc),
                'price': float(latest['Close']),
                'volume': int(latest['Volume']),
                'high': float(latest['High']),
                'low': float(latest['Low']),
                'open': float(latest['Open'])
            }

            # Insert directly to PostgreSQL
            cursor.execute("""
                INSERT INTO market_data (symbol, timestamp, price, volume, high, low, open, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                market_data['symbol'],
                market_data['timestamp'],
                market_data['price'],
                market_data['volume'],
                market_data['high'],
                market_data['low'],
                market_data['open'],
                datetime.utcnow()
            ))

            logger.info(f"Collected {symbol}: €{market_data['price']:.2f}")
            collected += 1

        except Exception as e:
            logger.error(f"Error collecting {symbol}: {e}")
            continue

    conn.commit()
    cursor.close()
    conn.close()

    logger.info(f"Collection complete: {collected}/{len(DAX_SYMBOLS)} symbols")
    return collected

def main():
    """Main collection loop"""
    logger.info("PostgreSQL Data Collector started")

    while True:
        try:
            start_time = time.time()
            count = collect_and_store_data()
            duration = time.time() - start_time

            logger.info(f"Collection cycle: {count} symbols in {duration:.2f}s")

            # Sleep for 5 minutes
            time.sleep(300)

        except KeyboardInterrupt:
            logger.info("Data collector stopped")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            time.sleep(60)

if __name__ == '__main__':
    main()