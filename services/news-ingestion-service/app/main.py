#!/usr/bin/env python3
"""
News & Economic Calendar Ingestion Service
Fetches real-time financial news and events for sentiment analysis
"""
import asyncio
import aiohttp
import pandas as pd
import psycopg2
from datetime import datetime, timedelta
import logging
import os
import json
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsIngestionService:
    def __init__(self):
        self.db_config = {
            'host': os.getenv('POSTGRES_HOST', 'postgres'),
            'port': int(os.getenv('POSTGRES_PORT', 5432)),
            'database': os.getenv('POSTGRES_DB', 'finrl_dax'),
            'user': os.getenv('POSTGRES_USER', 'finrl_user'),
            'password': os.getenv('POSTGRES_PASSWORD')
        }

        # News API endpoints (using free sources as fallback)
        self.news_sources = {
            'forex_factory': 'https://nfs.faireconomy.media/ff_calendar_thisweek.json',
            'alpha_vantage': f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=FOREX:EUR,FOREX:GBP&apikey={os.getenv('ALPHA_VANTAGE_KEY')}",
            'fmp': f"https://financialmodelingprep.com/api/v3/fmp/articles?page=0&size=50&apikey={os.getenv('FMP_KEY', 'demo')}"
        }

        self.currency_keywords = {
            'EURUSD': ['EUR', 'USD', 'ECB', 'Federal Reserve', 'Euro', 'Dollar', 'Eurozone', 'Lagarde', 'Powell'],
            'GBPUSD': ['GBP', 'USD', 'BOE', 'Federal Reserve', 'Pound', 'Dollar', 'Brexit', 'Bailey', 'Powell']
        }

    async def create_tables(self):
        """Create necessary tables for news and events"""
        conn = psycopg2.connect(**self.db_config)
        try:
            with conn.cursor() as cur:
                # News articles table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS news_articles (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMPTZ DEFAULT NOW(),
                        source VARCHAR(50),
                        title TEXT,
                        content TEXT,
                        url TEXT,
                        relevance_score DECIMAL(3,2),
                        currency_pair VARCHAR(10),
                        sentiment_score DECIMAL(4,3),
                        processed BOOLEAN DEFAULT FALSE
                    );
                """)

                # Economic events table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS economic_events (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMPTZ,
                        event_time TIMESTAMPTZ,
                        title VARCHAR(255),
                        country VARCHAR(10),
                        currency VARCHAR(3),
                        impact VARCHAR(20), -- High, Medium, Low
                        forecast VARCHAR(50),
                        previous VARCHAR(50),
                        actual VARCHAR(50),
                        processed BOOLEAN DEFAULT FALSE
                    );
                """)

                # Indexes for performance
                cur.execute("CREATE INDEX IF NOT EXISTS idx_news_timestamp ON news_articles(timestamp);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_news_processed ON news_articles(processed);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_events_time ON economic_events(event_time);")

                conn.commit()
                logger.info("Tables created successfully")
        finally:
            conn.close()

    async def fetch_forex_factory_calendar(self) -> List[Dict]:
        """Fetch economic calendar from Forex Factory"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.news_sources['forex_factory']) as response:
                    if response.status == 200:
                        events = await response.json()
                        return self.parse_forex_factory_events(events)
        except Exception as e:
            logger.error(f"Error fetching Forex Factory calendar: {e}")
        return []

    def parse_forex_factory_events(self, events: List[Dict]) -> List[Dict]:
        """Parse Forex Factory events into standardized format"""
        parsed_events = []
        for event in events:
            try:
                parsed_event = {
                    'timestamp': datetime.now(),
                    'event_time': datetime.fromisoformat(event.get('date', '')),
                    'title': event.get('title', ''),
                    'country': event.get('country', ''),
                    'currency': event.get('currency', ''),
                    'impact': event.get('impact', 'Low'),
                    'forecast': event.get('forecast', ''),
                    'previous': event.get('previous', ''),
                    'actual': event.get('actual', '')
                }
                parsed_events.append(parsed_event)
            except Exception as e:
                logger.warning(f"Failed to parse event: {e}")
        return parsed_events

    async def fetch_financial_news(self) -> List[Dict]:
        """Fetch financial news from multiple sources"""
        all_articles = []

        # Simulate news fetching (replace with real APIs)
        sample_news = [
            {
                'source': 'reuters',
                'title': 'ECB signals dovish stance on monetary policy',
                'content': 'European Central Bank maintains accommodative monetary policy...',
                'url': 'https://reuters.com/sample',
                'timestamp': datetime.now() - timedelta(minutes=30)
            },
            {
                'source': 'bloomberg',
                'title': 'US Dollar strengthens amid Fed hawkish comments',
                'content': 'Federal Reserve officials suggest more aggressive tightening...',
                'url': 'https://bloomberg.com/sample',
                'timestamp': datetime.now() - timedelta(minutes=15)
            }
        ]

        for article in sample_news:
            # Calculate relevance score based on keywords
            relevance_scores = {}
            for pair, keywords in self.currency_keywords.items():
                score = sum(1 for keyword in keywords
                           if keyword.lower() in article['title'].lower() + ' ' + article['content'].lower())
                relevance_scores[pair] = min(score / len(keywords), 1.0)

            # Add articles for relevant currency pairs
            for pair, score in relevance_scores.items():
                if score > 0.1:  # Minimum relevance threshold
                    article_copy = article.copy()
                    article_copy['currency_pair'] = pair
                    article_copy['relevance_score'] = score
                    all_articles.append(article_copy)

        return all_articles

    async def store_news(self, articles: List[Dict]):
        """Store news articles in PostgreSQL"""
        if not articles:
            return

        conn = psycopg2.connect(**self.db_config)
        try:
            with conn.cursor() as cur:
                for article in articles:
                    cur.execute("""
                        INSERT INTO news_articles
                        (source, title, content, url, relevance_score, currency_pair, timestamp)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        article['source'],
                        article['title'],
                        article['content'],
                        article['url'],
                        article['relevance_score'],
                        article['currency_pair'],
                        article['timestamp']
                    ))
                conn.commit()
                logger.info(f"Stored {len(articles)} news articles")
        finally:
            conn.close()

    async def store_events(self, events: List[Dict]):
        """Store economic events in PostgreSQL"""
        if not events:
            return

        conn = psycopg2.connect(**self.db_config)
        try:
            with conn.cursor() as cur:
                for event in events:
                    cur.execute("""
                        INSERT INTO economic_events
                        (event_time, title, country, currency, impact, forecast, previous, actual)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT DO NOTHING
                    """, (
                        event['event_time'],
                        event['title'],
                        event['country'],
                        event['currency'],
                        event['impact'],
                        event['forecast'],
                        event['previous'],
                        event['actual']
                    ))
                conn.commit()
                logger.info(f"Stored {len(events)} economic events")
        finally:
            conn.close()

    async def run_ingestion_cycle(self):
        """Run one complete ingestion cycle"""
        logger.info("Starting news ingestion cycle")

        # Fetch and store news
        articles = await self.fetch_financial_news()
        await self.store_news(articles)

        # Fetch and store economic events
        events = await self.fetch_forex_factory_calendar()
        await self.store_events(events)

        logger.info("Ingestion cycle completed")

    async def run_continuous_ingestion(self, interval_minutes: int = 5):
        """Run continuous news ingestion"""
        await self.create_tables()

        while True:
            try:
                await self.run_ingestion_cycle()
                await asyncio.sleep(interval_minutes * 60)
            except Exception as e:
                logger.error(f"Error in ingestion cycle: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry

if __name__ == "__main__":
    service = NewsIngestionService()
    asyncio.run(service.run_continuous_ingestion())