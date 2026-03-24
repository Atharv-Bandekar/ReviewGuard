from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import os
import tensorflow as tf
from transformers import DebertaV2Tokenizer, TFDebertaV2ForSequenceClassification
import traceback
import threading
import re
import math

# Import XAI Service
from backend.xai_service import get_explanation, stream_explanation

# --- CONFIGURATION ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Force CPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Reduce log spam

app = Flask(__name__)
CORS(app)

BASE = os.path.dirname(__file__)
MODEL_DIR_AMAZON = os.path.join(BASE, 'backend', 'model')          # DeBERTa

# --- GLOBAL VARIABLES ---
amazon_model = None
amazon_tokenizer = None

# --- LOADER FUNCTIONS ---
def preload_amazon_model():
    global amazon_model, amazon_tokenizer
    print("---------------------------------------------------------------")
    print("Loading Amazon DeBERTa Model (Fraud-Style Axis)...")
    print("---------------------------------------------------------------")
    try:
        amazon_tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_DIR_AMAZON)
        amazon_model = TFDebertaV2ForSequenceClassification.from_pretrained(MODEL_DIR_AMAZON)
        print("Amazon model loaded successfully.")
    except Exception as e:
        print(f"Error loading Amazon model: {e}")

# --- AXIS 2: HEURISTICS ENGINE (AI vs Human) ---
def calculate_ai_likelihood(text):
    """Dynamically calculates probability of AI generation based on linguistic heuristics"""
    if not text or len(text.strip()) == 0:
        return 0.0

    score = 0.0
    
    # Extract purely alphabetic words
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    if len(words) < 5: return 0.0
    
    # ---------------------------------------------------------
    # 1. LEXICAL COMPLEXITY (Replaces Hardcoded Buzzwords)
    # ---------------------------------------------------------
    # Humans write conversational Amazon reviews (avg word length ~4.2 - 4.6 chars).
    # AI mathematically skews toward denser, academic text (avg 5.2 - 6.0+ chars).
    
    complex_words = sum(1 for w in words if len(w) >= 7) # Proxy for 3+ syllables
    complex_word_ratio = complex_words / len(words)
    avg_word_length = sum(len(w) for w in words) / len(words)

    # If the text is dense with complex vocabulary, flag it
    if avg_word_length > 5.5 and complex_word_ratio > 0.25:
        score += 0.35  # Heavy penalty for highly academic structure
    elif avg_word_length > 5.0 and complex_word_ratio > 0.18:
        score += 0.15
        
    # ---------------------------------------------------------
    # 2. SENTENCE LENGTH UNIFORMITY
    # ---------------------------------------------------------
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if len(sentences) > 1:
        lengths = [len(s.split()) for s in sentences]
        mean_len = sum(lengths) / len(lengths)
        variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
        
        # AI creates highly uniform, perfectly paced sentences (low variance)
        if variance < 10 and mean_len > 12: 
            score += 0.35
        elif variance < 20:
            score += 0.15
            
    # ---------------------------------------------------------
    # 3. TYPE-TOKEN RATIO (Repetitiveness)
    # ---------------------------------------------------------
    unique_words = len(set(words))
    ttr = unique_words / len(words)
    
    # AI repeats structural words more often than humans
    if len(words) > 40 and ttr < 0.45:
        score += 0.2 
        
    return min(score, 0.95) # Cap at 95% certainty for heuristics

# --- HELPER: PROCESS PREDICTIONS ---
def get_combined_assessment(fake_score, real_score, ai_score):
    # Axis 1: Fraud Style (DeBERTa)
    if fake_score > 0.60:
        style = "Promotional-style"
        style_conf = fake_score
    elif real_score > 0.60:
        style = "Genuine-style"
        style_conf = real_score
    else:
        style = "Uncertain-style"
        style_conf = max(fake_score, real_score)
        
    # Axis 2: Authorship (Heuristics)
    if ai_score >= 0.50:
        author = "AI-assisted"
        author_conf = ai_score
    else:
        author = "Human-written"
        author_conf = 1.0 - ai_score # Inverse score for human confidence

    # Final Combined Label & Averaged Confidence
    combined_label = f"{style}, {author}"
    final_conf = (style_conf + author_conf) / 2.0
    
    # Strictly cap the maximum confidence at 0.99 (99%)
    final_conf = min(final_conf, 0.99)
    
    return combined_label, final_conf

# --- ROUTES ---

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    if not amazon_model: return jsonify({'error': 'Amazon Model is loading.'}), 503
    try:
        data = request.json
        texts = data.get('texts', [])
        if not texts: return jsonify({'results': []})

        # Axis 1: DeBERTa Inference
        inputs = amazon_tokenizer(texts, return_tensors="tf", truncation=True, padding=True, max_length=128)
        logits = amazon_model(inputs).logits
        probs_batch = tf.nn.softmax(logits, axis=1).numpy()
        
        results = []
        for i, text in enumerate(texts):
            probs = probs_batch[i]
            fake_score = float(probs[1])
            real_score = float(probs[0])
            
            # Axis 2: Heuristics Inference
            ai_score = calculate_ai_likelihood(text)
            
            label, conf = get_combined_assessment(fake_score, real_score, ai_score)
            
            scores = {
                "fraud_style_score": fake_score, 
                "genuine_style_score": real_score,
                "ai_likelihood_score": ai_score
            }
            results.append({'label': label, 'confidence': conf, 'scores': scores})

        return jsonify({'results': results})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict_review():
    if not amazon_model: return jsonify({'error': 'Amazon Model is loading.'}), 503
    try:
        data = request.json
        text = data.get('text', '')
        
        # Axis 1
        inputs = amazon_tokenizer([text], return_tensors="tf", truncation=True, padding=True, max_length=128)
        logits = amazon_model(inputs).logits
        probs = tf.nn.softmax(logits, axis=1).numpy()[0]
        
        # Axis 2
        ai_score = calculate_ai_likelihood(text)
        
        label, conf = get_combined_assessment(float(probs[1]), float(probs[0]), ai_score)
        
        scores = {
            "fraud_style_score": float(probs[1]), 
            "genuine_style_score": float(probs[0]),
            "ai_likelihood_score": ai_score
        }
        
        return jsonify({'label': label, 'confidence': conf, 'scores': scores})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# STREAMING EXPLANATION ENDPOINT
@app.route('/explain_stream', methods=['GET'])
def explain_stream():
    text = request.args.get('text', '')
    label = request.args.get('label', 'UNKNOWN')
    confidence = float(request.args.get('confidence', 0))

    def generate():
        for chunk in stream_explanation(text, label, confidence):
            if chunk: yield chunk

    return Response(stream_with_context(generate()), mimetype='text/plain')

# STANDARD EXPLANATION ENDPOINT (Fallback)
@app.route('/explain', methods=['POST'])
def explain():
    try:
        data = request.json
        label = data.get('label', 'UNKNOWN')
        return jsonify({'explanation': get_explanation(data.get('text', ''), label, float(data.get('confidence', 0)))})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    threading.Thread(target=preload_amazon_model).start()
    app.run(host='0.0.0.0', port=8000, debug=True, use_reloader=False)