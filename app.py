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

# --- LOADER FUNCTIONS ---
def preload_amazon_model():
    global amazon_model, amazon_tokenizer
    print("---------------------------------------------------------------")
    print("⏳ INITIALIZING: Loading Amazon DeBERTa Model...")
    print("---------------------------------------------------------------")
    try:
        amazon_tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_DIR_AMAZON)
        amazon_model = TFDebertaV2ForSequenceClassification.from_pretrained(MODEL_DIR_AMAZON)
        print("✅ AMAZON MODEL READY!")
    except Exception as e:
        print(f"❌ Critical Error loading Amazon model: {e}")

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

# --- HELPER: PROCESS PREDICTION ---
def process_amazon_probs(probs):
    fake_score = float(probs[1])
    real_score = float(probs[0])
    
    if fake_score > 0.60:
        label = "FAKE"
        conf = fake_score
    elif real_score > 0.60:
        label = "GENUINE"
        conf = real_score
    else:
        label = "UNCERTAIN"
        conf = max(fake_score, real_score)
    
    # CAP CONFIDENCE AT 99%
    return label, min(conf, 0.99)

def process_social_probs(probs):
    bot_score = float(probs[1])
    human_score = float(probs[0])
    
    if bot_score > 0.70:
        label = "AI"
        conf = bot_score
    else:
        label = "HUMAN"
        conf = human_score
        
    # CAP CONFIDENCE AT 99%
    return label, min(conf, 0.99)

# --- ROUTES ---

# 1. BATCH ROUTE (For Content Script / Auto-Scan)
@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    if not amazon_model: return jsonify({'error': 'Amazon Model is loading.'}), 503
    try:
        data = request.json
        texts = data.get('texts', [])
        if not texts: return jsonify({'results': []})

        inputs = amazon_tokenizer(texts, return_tensors="tf", truncation=True, padding=True, max_length=128)
        logits = amazon_model(inputs).logits
        probs_batch = tf.nn.softmax(logits, axis=1).numpy()
        
        results = []
        for probs in probs_batch:
            label, conf = process_amazon_probs(probs)
            results.append({'label': label, 'confidence': conf})

        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 2. SINGLE ROUTE (For Popup / Manual Check) -- FIXES "Undefined NaN"
@app.route('/predict', methods=['POST'])
def predict_review():
    if not amazon_model: return jsonify({'error': 'Amazon Model is loading.'}), 503
    try:
        data = request.json
        text = data.get('text', '')
        
        # Wrap in list for batch-style processing, but unwrap result
        inputs = amazon_tokenizer([text], return_tensors="tf", truncation=True, padding=True, max_length=128)
        logits = amazon_model(inputs).logits
        probs = tf.nn.softmax(logits, axis=1).numpy()[0]
        
        label, conf = process_amazon_probs(probs)
        
        # RETURN SINGLE OBJECT (Not a list)
        return jsonify({'label': label, 'confidence': conf})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 3. SINGLE COMMENT ROUTE (For Popup / Manual Check)
@app.route('/predict_comment', methods=['POST'])
def predict_comment():
    model, tokenizer = get_social_model()
    if not model: return jsonify({'error': 'Social Model missing'}), 500
    try:
        data = request.json
        text = data.get('text', '')
        
        inputs = tokenizer([text], return_tensors="tf", truncation=True, padding=True, max_length=128)
        logits = model(inputs).logits
        probs = tf.nn.softmax(logits, axis=1).numpy()[0]
        
        label, conf = process_social_probs(probs)
        
        return jsonify({'label': label, 'confidence': conf})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 4. SOCIAL BATCH (Optional, for future)
@app.route('/predict_comment_batch', methods=['POST'])
def predict_comment_batch():
    model, tokenizer = get_social_model()
    if not model: return jsonify({'error': 'Social Model missing'}), 500
    try:
        data = request.json
        texts = data.get('texts', [])
        inputs = tokenizer(texts, return_tensors="tf", truncation=True, padding=True, max_length=128)
        logits = model(inputs).logits
        probs_batch = tf.nn.softmax(logits, axis=1).numpy()
        
        results = []
        for probs in probs_batch:
            label, conf = process_social_probs(probs)
            results.append({'label': label, 'confidence': conf})
            
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/explain', methods=['POST'])
def explain():
    try:
        data = request.json
        # Handle the empty label error safely
        label = data.get('label', 'UNKNOWN')
        return jsonify({'explanation': get_explanation(data.get('text', ''), label, float(data.get('confidence', 0)))})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    threading.Thread(target=preload_amazon_model).start()
    app.run(host='0.0.0.0', port=8000, debug=True, use_reloader=False)