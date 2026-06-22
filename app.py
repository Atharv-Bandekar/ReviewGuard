"""
app.py — E-commerce Review Detector (TensorFlow + Groq Attention XAI)

Upgrades:
  - Integrates load_explainer to pass the TF model to the Attention XAI service.
  - Removed all social media / TinyBERT logic to focus strictly on Amazon reviews.
  - Implemented MAX_LEN = 128 (matches your training configuration).
  - All endpoints upgraded to POST to handle massive reviews without HTTP 414 errors.
  - Integrates the robust heuristics engine (Coefficient of Variation, 3-tier labels).
  - Temperature Calibration ready (CALIBRATION_T).
  - Cleaned for Production/Demo: Removed all debug and shadow run artifacts.
"""

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import os
import tensorflow as tf
import numpy as np
from transformers import DebertaV2Tokenizer, TFDebertaV2ForSequenceClassification
import threading
import re

from backend.xai_service import stream_explanation, load_explainer
from backend.heuristics_engine import analyze_text_heuristics

# ── CONFIGURATION ──────────────────────────────────────────────────────
MAX_LEN = 128

# If you haven't run temperature calibration, leave at 1.0 (no-op)
CALIBRATION_T = 1.2375  

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Force CPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Reduce log spam

app = Flask(__name__)
CORS(app)

BASE      = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE, "backend", "model")

amazon_model     = None
amazon_tokenizer = None

def preload_model():
    global amazon_model, amazon_tokenizer
    print("\n---------------------------------------------------------------")
    print("Loading Amazon DeBERTa Model (TensorFlow)...")
    print("---------------------------------------------------------------")
    try:
        amazon_tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_DIR)
        amazon_model     = TFDebertaV2ForSequenceClassification.from_pretrained(MODEL_DIR)
        print("Amazon model loaded successfully.")

        load_explainer(amazon_model, amazon_tokenizer)

    except Exception as e:
        print(f"Error loading Amazon model: {e}")

# ── CALIBRATED ASSESSMENT ──────────────────────────────────────────────

def _calibrated_probs(raw_fake: float, raw_real: float) -> tuple[float, float]:
    """Apply temperature scaling to raw softmax outputs."""
    if abs(CALIBRATION_T - 1.0) < 1e-4:
        return raw_fake, raw_real

    log_odds        = np.log(raw_fake / (raw_real + 1e-9))
    scaled_log_odds = log_odds / CALIBRATION_T
    fake_cal        = float(1.0 / (1.0 + np.exp(-scaled_log_odds)))
    return fake_cal, 1.0 - fake_cal

def get_combined_assessment(raw_fake_prob: float,
                             raw_real_prob: float,
                             ai_score: float) -> tuple[str, float]:
    
    fake_prob, real_prob = _calibrated_probs(raw_fake_prob, raw_real_prob)

    # ── Axis 1: style (DeBERTa-owned) ──────────────────────────────
    # AGGRESSIVE THRESHOLD: Narrowed the "Uncertain" band to force a decision.
    FAKE_THRESHOLD    = 0.9000   # data-driven, shadow run      
    GENUINE_THRESHOLD = 0.4000   # asymmetric (FP cost > FN)   
                                                                        
    if fake_prob >= FAKE_THRESHOLD:                                    
         style      = "Promotional-style"                               
         style_conf = min(fake_prob, 0.98)                              
    elif fake_prob <= GENUINE_THRESHOLD:                               
         style      = "Genuine-style"                                   
         style_conf = min(1.0 - fake_prob, 0.98)                        
    else:                                                              
         style      = "Uncertain-style"                                 
         style_conf = max(fake_prob, 1.0 - fake_prob)

    # ── Axis 2: authorship (Neutralized) ───────────────────────────
    # We agreed rule-based AI detection causes false positives for terse humans.
    # We force "Human-written" to safely bypass the AI heuristics while 
    # keeping the frontend popup.js 2-part string matching perfectly intact.
    author = "Human-written"

    AI_THRESHOLD = 0.60

    if ai_score >= AI_THRESHOLD:
        author = "AI-assisted"
    else:
        author = "Human-written"

    combined_label = f"{style}, {author}"
    final_conf = round(style_conf, 4)

    return combined_label, final_conf

def clean_live_inference_text(text):
    """
    Defense-in-depth: Strips lingering Amazon DOM artifacts 
    just in case the frontend scraper misses them.
    """
    if not text:
        return ""
    text = str(text)
    
    # Remove review metadata patterns
    text = re.sub(r'\d+\.\d+ out of 5 stars', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Reviewed in [a-zA-Z\s]+ on \d{1,2} [a-zA-Z]+ \d{4}', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Verified Purchase', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Colour:\s*[a-zA-Z0-9\s]+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Size:\s*[a-zA-Z0-9\s]+', '', text, flags=re.IGNORECASE)
    
    # Remove lingering helpful button text
    text = re.sub(r'\d+ people found this helpful', '', text, flags=re.IGNORECASE)
    text = re.sub(r'One person found this helpful', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bHelpful\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bReport\b', '', text, flags=re.IGNORECASE)
    
    # Clean up massive whitespace gaps
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s{2,}', ' ', text)
    
    return text.strip()

# ── ROUTES ────────────────────────────────────────────────────────────

@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    if not amazon_model:
        return jsonify({"error": "Amazon Model is loading, please retry."}), 503
    try:
        data  = request.json
        texts = data.get("texts", [])
        if not texts:
            return jsonify({"results": []})

        valid_texts, valid_indices = [], []
        for i, t in enumerate(texts):
            # Clean the text BEFORE validating its length
            cleaned_t = clean_live_inference_text(t)
            if cleaned_t and len(cleaned_t.split()) >= 5:
                valid_texts.append(cleaned_t)
                valid_indices.append(i)
        
        results = [None] * len(texts)

        if valid_texts:
            inputs = amazon_tokenizer(
                valid_texts, return_tensors="tf",
                truncation=True, padding=True, max_length=MAX_LEN
            )
            logits = amazon_model(inputs).logits
            probs_batch = tf.nn.softmax(logits, axis=1).numpy()

            for batch_idx, original_idx in enumerate(valid_indices):
                text     = valid_texts[batch_idx]
                raw_real = float(probs_batch[batch_idx][1])
                raw_fake = float(probs_batch[batch_idx][0])

                # Route through robust heuristics
                _, _, ai_score, _ = analyze_text_heuristics(text, raw_fake, raw_real)
                label, conf = get_combined_assessment(raw_fake, raw_real, ai_score)

                results[original_idx] = {
                    "label":      label,
                    "confidence": conf,
                    "scores": {
                        "fraud_style_score":   raw_fake,
                        "genuine_style_score": raw_real,
                        "ai_likelihood_score": ai_score,
                    },
                }

        # Fill in skipped (too-short) reviews
        for i, r in enumerate(results):
            if r is None:
                results[i] = {
                    "label":      "Uncertain-style, Human-written",
                    "confidence": 0.50,
                    "scores": {
                        "fraud_style_score":   0.50,
                        "genuine_style_score": 0.50,
                        "ai_likelihood_score": 0.0,
                    },
                }

        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict_review():
    if not amazon_model:
        return jsonify({"error": "Amazon Model is loading, please retry."}), 503
    try:
        data = request.json
        text = data.get("text", "").strip()

        if not text or len(text.split()) < 5:
            return jsonify({"error": "Text too short to analyze"}), 400

        inputs = amazon_tokenizer(
            [text], return_tensors="tf",
            truncation=True, padding=True, max_length=MAX_LEN
        )
        logits = amazon_model(inputs).logits
        probs  = tf.nn.softmax(logits, axis=1).numpy()[0]
        
        raw_real = float(probs[1])
        raw_fake = float(probs[0])

        # Route through robust heuristics
        _, _, ai_score, _ = analyze_text_heuristics(text, raw_fake, raw_real)
        label, conf       = get_combined_assessment(raw_fake, raw_real, ai_score)

        return jsonify({
            "label":      label,
            "confidence": conf,
            "scores": {
                "fraud_style_score":   raw_fake,
                "genuine_style_score": raw_real,
                "ai_likelihood_score": ai_score,
            },
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── EXPLAINABLE AI ROUTE (POST) ──
@app.route("/explain_stream", methods=["POST"])
def explain_stream():
    data       = request.json
    # Clean the text before XAI processes it
    text       = clean_live_inference_text(data.get("text", ""))
    label      = data.get("label", "UNKNOWN")
    confidence = float(data.get("confidence", 0))

    if not text:
        return Response("Error: No text provided.", mimetype="text/plain")

    def generate():
        for chunk in stream_explanation(text, label, confidence):
            if chunk:
                yield chunk

    response = Response(stream_with_context(generate()), mimetype="text/plain")
    response.headers["Cache-Control"]     = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"]            = "no-cache"
    response.headers["Expires"]           = "0"
    response.headers["X-Accel-Buffering"] = "no"
    return response

if __name__ == "__main__":
    print("\n=== INITIATING REVIEWGUARD BOOT SEQUENCE ===")
    threading.Thread(target=preload_model, daemon=True).start()
    app.run(host="0.0.0.0", port=8000, debug=True, use_reloader=False)