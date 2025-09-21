from flask import Flask, request, jsonify
import json
import logging

# Simple sentiment analyzer
def simple_sentiment(text):
    positive_words = ['up', 'rise', 'gain', 'profit', 'growth', 'strong', 'good', 'excellent', 'positive', 'bullish', 'buy']
    negative_words = ['down', 'fall', 'loss', 'decline', 'weak', 'bad', 'poor', 'negative', 'bearish', 'sell']

    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)

    if pos_count > neg_count:
        return {"label": "POSITIVE", "score": 0.7 + (pos_count * 0.1)}
    elif neg_count > pos_count:
        return {"label": "NEGATIVE", "score": 0.7 + (neg_count * 0.1)}
    else:
        return {"label": "NEUTRAL", "score": 0.5}

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.route("/health")
def health():
    return {"status": "healthy", "model": "simple-sentiment"}

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    text = data.get("text", "")
    result = simple_sentiment(text)
    return jsonify({"text": text, "sentiment": result["label"], "confidence": result["score"]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
