from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tensorflow as tf
from transformers import DebertaV2Tokenizer, TFDebertaV2ForSequenceClassification, AutoTokenizer, TFAutoModelForSequenceClassification
import traceback
import threading

# Import XAI Service
from backend.xai_service import get_explanation 

# --- CONFIGURATION ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Force CPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Reduce log spam

app = Flask(__name__)
CORS(app)

BASE = os.path.dirname(__file__)
MODEL_DIR_AMAZON = os.path.join(BASE, 'backend', 'model')          # DeBERTa
MODEL_DIR_SOCIAL = os.path.join(BASE, 'backend', 'model_tinybert') # TinyBERT

# --- GLOBAL VARIABLES ---
amazon_model = None
amazon_tokenizer = None
social_model = None
social_tokenizer = None

# --- 1. EAGER LOADER (AMAZON) ---
# We run this immediately when the app starts
def preload_amazon_model():
    global amazon_model, amazon_tokenizer
    print("---------------------------------------------------------------")
    print("⏳ INITIALIZING: Loading Amazon DeBERTa Model... (Please Wait)")
    print("   This may take 15-20 seconds, but scans will be fast later.")
    print("---------------------------------------------------------------")
    try:
        amazon_tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_DIR_AMAZON)
        amazon_model = TFDebertaV2ForSequenceClassification.from_pretrained(MODEL_DIR_AMAZON)
        print("✅ AMAZON MODEL READY! (Server is fully operational)")
    except Exception as e:
        print(f"❌ Critical Error loading Amazon model: {e}")

# --- 2. LAZY LOADER (SOCIAL) ---
# We run this only when a YouTube/X request hits
def get_social_model():
    global social_model, social_tokenizer
    if social_model is None:
        print(f"⏳ LAZY LOADING: Loading Social TinyBERT...")
        try:
            social_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR_SOCIAL)
            social_model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_DIR_SOCIAL)
            print("✅ SOCIAL MODEL READY!")
        except Exception as e:
            print(f"❌ Failed to load Social Model: {e}")
            return None, None
    return social_model, social_tokenizer

# --- ROUTES ---

@app.route('/predict', methods=['POST'])
def predict_review():
    # Use the pre-loaded model
    if not amazon_model: 
        return jsonify({'error': 'Amazon Model is still loading or failed.'}), 503
    
    try:
        data = request.json
        text = data.get('text', '')
        if not text: return jsonify({'error': 'No text'}), 400

        inputs = amazon_tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=128)
        logits = amazon_model(inputs).logits
        probs = tf.nn.softmax(logits, axis=1).numpy()[0]
        
        real_score = float(probs[0])
        fake_score = float(probs[1])
        
        if fake_score > 0.60:
            label = "FAKE"
            confidence = fake_score
            explanation = "DeBERTa detected AI patterns."
        elif real_score > 0.60:
            label = "GENUINE"
            confidence = real_score
            explanation = "DeBERTa detected human patterns."
        else:
            label = "UNCERTAIN"
            confidence = max(fake_score, real_score)
            explanation = "Mixed signals."

        return jsonify({'label': label, 'confidence': min(confidence, 0.99), 'explanation': explanation})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_comment', methods=['POST'])
def predict_comment():
    # Trigger lazy load
    model, tokenizer = get_social_model() 
    if not model: return jsonify({'error': 'Social Model missing'}), 500

    try:
        data = request.json
        text = data.get('text', '')
        
        inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=128)
        logits = model(inputs).logits
        probs = tf.nn.softmax(logits, axis=1).numpy()[0]
        
        human_score = float(probs[0])
        bot_score = float(probs[1]) 

        if bot_score > 0.70:
            label = "AI"
            confidence = bot_score
        else:
            label = "HUMAN"
            confidence = human_score

        print(f"[SOCIAL] {label} ({confidence:.2f})")
        return jsonify({'label': label, 'confidence': confidence})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/explain', methods=['POST'])
def explain():
    try:
        data = request.json
        return jsonify({'explanation': get_explanation(data.get('text', ''), data.get('label', ''), float(data.get('confidence', 0)))})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # We use a thread to load the model so the server starts accepting requests immediately
    # (Though requests will fail with 503 until the thread finishes)
    threading.Thread(target=preload_amazon_model).start()
    
    app.run(host='0.0.0.0', port=8000, debug=True, use_reloader=False)