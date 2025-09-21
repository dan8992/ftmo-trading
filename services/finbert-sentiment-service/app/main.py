#!/usr/bin/env python3
"""
FinBERT Sentiment Analysis Service
Processes financial news and calculates sentiment scores for trading signals
"""
import asyncio
import psycopg2
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import logging
import os
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinBERTSentimentService:
    def __init__(self):
        self.db_config = {
            'host': os.getenv('POSTGRES_HOST', 'postgres'),
            'port': int(os.getenv('POSTGRES_PORT', 5432)),
            'database': os.getenv('POSTGRES_DB', 'finrl_dax'),
            'user': os.getenv('POSTGRES_USER', 'finrl_user'),
            'password': os.getenv('POSTGRES_PASSWORD')
        }
        
        # Initialize FinBERT model
        self.model_name = "ProsusAI/finbert"
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Sentiment mapping
        self.label_mapping = {
            'positive': 1.0,
            'neutral': 0.0,
            'negative': -1.0
        }

    async def initialize_model(self):
        """Initialize FinBERT model and tokenizer"""
        try:
            logger.info("Loading FinBERT model...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"FinBERT model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load FinBERT model: {e}")
            # Fallback to simple sentiment scoring
            self.model = None
            self.tokenizer = None

    def analyze_sentiment(self, text: str) -> Tuple[float, float]:
        """
        Analyze sentiment of financial text
        Returns: (sentiment_score, confidence)
        """
        if self.model is None or self.tokenizer is None:
            return self.fallback_sentiment_analysis(text)
        
        try:
            # Tokenize and prepare input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
            # Extract sentiment score and confidence
            probs = predictions.cpu().numpy()[0]
            predicted_class = np.argmax(probs)
            confidence = float(probs[predicted_class])
            
            # Map to sentiment score (-1 to 1)
            class_labels = ['negative', 'neutral', 'positive']
            sentiment_label = class_labels[predicted_class]
            sentiment_score = self.label_mapping[sentiment_label]
            
            # Adjust sentiment score by confidence
            if sentiment_label == 'neutral':
                sentiment_score = 0.0
            else:
                sentiment_score *= confidence
                
            return sentiment_score, confidence
            
        except Exception as e:
            logger.warning(f"FinBERT analysis failed, using fallback: {e}")
            return self.fallback_sentiment_analysis(text)

    def fallback_sentiment_analysis(self, text: str) -> Tuple[float, float]:
        """Simple keyword-based sentiment analysis as fallback"""
        positive_keywords = [
            'bullish', 'rally', 'gains', 'surge', 'strength', 'positive',
            'optimistic', 'growth', 'increase', 'rise', 'boost', 'support'
        ]
        
        negative_keywords = [
            'bearish', 'decline', 'falls', 'weakness', 'negative', 'crash',
            'pessimistic', 'recession', 'decrease', 'drop', 'pressure', 'risk'
        ]
        
        text_lower = text.lower()
        
        positive_count = sum(1 for word in positive_keywords if word in text_lower)
        negative_count = sum(1 for word in negative_keywords if word in text_lower)
        
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            return 0.0, 0.5  # Neutral sentiment, low confidence
        
        sentiment_score = (positive_count - negative_count) / max(total_sentiment_words, 1)
        confidence = min(total_sentiment_words / 10.0, 1.0)  # Max confidence at 10+ sentiment words
        
        return sentiment_score, confidence

    async def get_unprocessed_news(self, limit: int = 50) -> List[Dict]:
        """Fetch unprocessed news articles from database"""
        conn = psycopg2.connect(**self.db_config)
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, title, content, currency_pair, relevance_score
                    FROM news_articles 
                    WHERE processed = FALSE
                    ORDER BY timestamp DESC
                    LIMIT %s
                """, (limit,))
                
                columns = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
                
                return [dict(zip(columns, row)) for row in rows]
        finally:
            conn.close()

    async def update_sentiment_scores(self, article_id: int, sentiment_score: float, confidence: float):
        """Update sentiment score in database"""
        conn = psycopg2.connect(**self.db_config)
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE news_articles 
                    SET sentiment_score = %s, processed = TRUE
                    WHERE id = %s
                """, (sentiment_score, article_id))
                conn.commit()
        finally:
            conn.close()

    async def calculate_aggregate_sentiment(self, currency_pair: str, hours_back: int = 24) -> Dict:
        """Calculate aggregate sentiment for a currency pair over specified time period"""
        conn = psycopg2.connect(**self.db_config)
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        AVG(sentiment_score) as avg_sentiment,
                        COUNT(*) as article_count,
                        AVG(relevance_score) as avg_relevance,
                        STDDEV(sentiment_score) as sentiment_volatility
                    FROM news_articles 
                    WHERE currency_pair = %s 
                        AND processed = TRUE
                        AND timestamp > %s
                """, (currency_pair, datetime.now() - timedelta(hours=hours_back)))
                
                result = cur.fetchone()
                
                return {
                    'currency_pair': currency_pair,
                    'avg_sentiment': float(result[0]) if result[0] else 0.0,
                    'article_count': int(result[1]) if result[1] else 0,
                    'avg_relevance': float(result[2]) if result[2] else 0.0,
                    'sentiment_volatility': float(result[3]) if result[3] else 0.0,
                    'hours_back': hours_back,
                    'timestamp': datetime.now()
                }
        finally:
            conn.close()

    async def process_sentiment_batch(self):
        """Process a batch of unprocessed news articles"""
        logger.info("Processing sentiment analysis batch")
        
        articles = await self.get_unprocessed_news(limit=50)
        
        if not articles:
            logger.info("No unprocessed articles found")
            return
        
        processed_count = 0
        for article in articles:
            try:
                # Combine title and content for sentiment analysis
                text = f"{article['title']} {article['content']}"
                
                # Analyze sentiment
                sentiment_score, confidence = self.analyze_sentiment(text)
                
                # Weight by relevance
                weighted_sentiment = sentiment_score * article['relevance_score']
                
                # Update database
                await self.update_sentiment_scores(article['id'], weighted_sentiment, confidence)
                
                processed_count += 1
                
                logger.debug(f"Processed article {article['id']}: sentiment={weighted_sentiment:.3f}, confidence={confidence:.3f}")
                
            except Exception as e:
                logger.error(f"Error processing article {article['id']}: {e}")
        
        logger.info(f"Processed {processed_count} articles")

    async def get_current_sentiment_scores(self) -> Dict[str, Dict]:
        """Get current sentiment scores for all currency pairs"""
        currency_pairs = ['EURUSD', 'GBPUSD']
        sentiment_scores = {}
        
        for pair in currency_pairs:
            sentiment_data = await self.calculate_aggregate_sentiment(pair, hours_back=24)
            sentiment_scores[pair] = sentiment_data
            
        return sentiment_scores

    async def run_continuous_sentiment_analysis(self, interval_minutes: int = 2):
        """Run continuous sentiment analysis"""
        await self.initialize_model()
        
        while True:
            try:
                await self.process_sentiment_batch()
                await asyncio.sleep(interval_minutes * 60)
            except Exception as e:
                logger.error(f"Error in sentiment analysis cycle: {e}")
                await asyncio.sleep(60)

if __name__ == "__main__":
    service = FinBERTSentimentService()
    asyncio.run(service.run_continuous_sentiment_analysis())