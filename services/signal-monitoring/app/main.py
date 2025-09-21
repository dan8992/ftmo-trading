#!/usr/bin/env python3
"""
Signal Monitoring Service
Tracks signal performance, accuracy, and system health
"""
import asyncio
import psycopg2
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List
from flask import Flask, jsonify

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignalMonitoringService:
    def __init__(self):
        self.db_config = {
            'host': os.getenv('POSTGRES_HOST', 'postgres-service'),
            'port': int(os.getenv('POSTGRES_PORT', 5432)),
            'database': os.getenv('POSTGRES_DB', 'dax_trading'),
            'user': os.getenv('POSTGRES_USER', 'finrl_user'),
            'password': os.getenv('POSTGRES_PASSWORD')
        }

    def get_signal_performance(self) -> Dict:
        """Get signal performance metrics"""
        conn = psycopg2.connect(**self.db_config)
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        COUNT(*) as total_signals,
                        AVG(confidence) as avg_confidence,
                        COUNT(CASE WHEN confidence > 0.7 THEN 1 END) as high_confidence_signals
                    FROM trading_signals
                    WHERE timestamp > %s
                """, (datetime.now() - timedelta(hours=24),))

                result = cur.fetchone()
                return {
                    'total_signals': result[0],
                    'avg_confidence': float(result[1]) if result[1] else 0.0,
                    'high_confidence_signals': result[2],
                    'timestamp': datetime.now()
                }
        finally:
            conn.close()

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now()})

@app.route('/metrics')
def metrics():
    service = SignalMonitoringService()
    return jsonify(service.get_signal_performance())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

