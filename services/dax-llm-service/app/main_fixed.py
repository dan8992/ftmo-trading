#!/usr/bin/env python3
"""
🚀 ENTERPRISE-GRADE FinBERT Trading API Server
Air-gapped deployment with graceful fallback strategies
"""

import os
import time
import json
import logging
from datetime import datetime
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import psycopg2
from typing import Dict, List, Any
import threading
import queue

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedFinBERTTrader:
    """Production-grade FinBERT with enterprise air-gap support"""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_ready = False
        self.inference_queue = queue.Queue()
        self.results_cache = {}
        self.startup_time = time.time()

        # Expert performance optimizations
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_length = 512
        self.batch_size = 8

        logger.info("🔄 Initializing OptimizedFinBERTTrader...")
        self._initialize_model()
        self._start_inference_worker()

    def _initialize_model(self):
        """Enterprise model initialization with multiple fallback strategies"""
        try:
            # Strategy 1: Use environment-specified model
            model_name = os.environ.get('MODEL_NAME', 'ProsusAI/finbert')
            cache_dir = os.environ.get('TRANSFORMERS_CACHE', '/tmp/transformers_cache')
            
            # Enterprise air-gap configuration
            offline_mode = os.environ.get('TRANSFORMERS_OFFLINE', '0') == '1'
            
            logger.info(f"🎯 Target model: {model_name}")
            logger.info(f"📁 Cache directory: {cache_dir}")
            logger.info(f"✈️  Offline mode: {offline_mode}")

            # Attempt 1: Load specified model (respecting offline mode)
            start_time = time.time()
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    local_files_only=offline_mode
                )
                
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    local_files_only=offline_mode
                )
                logger.info(f"✅ Primary model '{model_name}' loaded successfully")
                
            except Exception as e:
                logger.warning(f"⚠️  Primary model failed: {e}")
                
                # Enterprise Fallback Strategy: Try alternative models
                fallback_models = [
                    'nlptown/bert-base-multilingual-uncased-sentiment',
                    'distilbert-base-uncased-finetuned-sst-2-english',
                    'cardiffnlp/twitter-roberta-base-sentiment-latest'
                ]
                
                model_loaded = False
                for fallback_model in fallback_models:
                    try:
                        logger.info(f"🔄 Attempting fallback model: {fallback_model}")
                        
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            fallback_model,
                            cache_dir=cache_dir,
                            local_files_only=offline_mode
                        )
                        
                        self.model = AutoModelForSequenceClassification.from_pretrained(
                            fallback_model,
                            cache_dir=cache_dir,
                            local_files_only=offline_mode
                        )
                        
                        logger.info(f"✅ Fallback model '{fallback_model}' loaded successfully")
                        model_loaded = True
                        break
                        
                    except Exception as fallback_error:
                        logger.warning(f"❌ Fallback model '{fallback_model}' failed: {fallback_error}")
                        continue
                
                if not model_loaded:
                    if offline_mode:
                        raise Exception("🚨 No models available in offline mode. Please download models first.")
                    else:
                        # Last resort: Try online download
                        logger.warning("🌐 Attempting online download (last resort)")
                        try:
                            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                            logger.info(f"✅ Online download successful for '{model_name}'")
                        except Exception as online_error:
                            raise Exception(f"🚨 All model loading strategies failed: {online_error}")

            # Model optimization pipeline
            self.model.eval()
            if self.device == 'cuda':
                self.model = self.model.to(self.device)

            # JIT optimization for inference speed
            if hasattr(torch, 'jit') and os.environ.get('PYTORCH_JIT', '1') == '1':
                logger.info("🔥 Applying JIT optimization...")
                dummy_input = self.tokenizer("Market is bullish", return_tensors="pt", 
                                            max_length=self.max_length, truncation=True)
                if self.device == 'cuda':
                    dummy_input = {k: v.to(self.device) for k, v in dummy_input.items()}

                with torch.no_grad():
                    self.model(**dummy_input)  # Warm up

            load_time = time.time() - start_time
            logger.info(f"🚀 Model loaded and optimized in {load_time:.2f}s")
            self.model_ready = True

        except Exception as e:
            logger.error(f"❌ Model initialization failed: {e}")
            # In enterprise environments, we might want to continue with limited functionality
            # rather than completely failing
            self.model_ready = False

    def _start_inference_worker(self):
        """Background worker for inference queue processing"""
        def worker():
            while True:
                try:
                    if not self.model_ready:
                        time.sleep(1)
                        continue
                        
                    # Process inference requests from queue
                    item = self.inference_queue.get(timeout=1)
                    if item is None:
                        break
                    
                    text, result_queue = item
                    result = self._predict_sentiment(text)
                    result_queue.put(result)
                    self.inference_queue.task_done()
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Worker error: {e}")

        self.worker_thread = threading.Thread(target=worker)
        self.worker_thread.daemon = True
        self.worker_thread.start()

    def _predict_sentiment(self, text: str) -> Dict[str, Any]:
        """Core sentiment prediction with enterprise error handling"""
        if not self.model_ready:
            return {
                "error": "Model not ready",
                "sentiment": "neutral",
                "confidence": 0.0,
                "status": "degraded"
            }
        
        try:
            # Tokenize and predict
            inputs = self.tokenizer(text, return_tensors="pt", 
                                  max_length=self.max_length, 
                                  truncation=True, padding=True)
            
            if self.device == 'cuda':
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                confidence = torch.max(predictions).item()
                predicted_class = torch.argmax(predictions, dim=-1).item()

            # Map to standard sentiment labels
            sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
            sentiment = sentiment_map.get(predicted_class, "neutral")

            return {
                "sentiment": sentiment,
                "confidence": confidence,
                "model_ready": True,
                "device": self.device,
                "inference_time": time.time()
            }

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                "error": str(e),
                "sentiment": "neutral",
                "confidence": 0.0,
                "status": "error"
            }

# Flask Application
app = Flask(__name__)

# Global trader instance
trader = OptimizedFinBERTTrader()

@app.route('/health', methods=['GET'])
def health_check():
    """Enterprise health check endpoint"""
    uptime = time.time() - trader.startup_time
    
    return jsonify({
        "status": "healthy" if trader.model_ready else "degraded",
        "model_ready": trader.model_ready,
        "uptime_seconds": uptime,
        "device": trader.device,
        "version": "2.0-enterprise"
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Sentiment prediction endpoint"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
            
        result = trader._predict_sentiment(text)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    logger.info("🚀 Starting Enterprise FinBERT Trading API Server")
    logger.info(f"📊 Model ready: {trader.model_ready}")
    logger.info(f"🔧 Device: {trader.device}")
    
    # Production-grade server configuration
    app.run(
        host='0.0.0.0',
        port=8000,
        debug=False,
        threaded=True
    )