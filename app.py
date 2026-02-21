from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import os
import tensorflow as tf
from transformers import DebertaV2Tokenizer, TFDebertaV2ForSequenceClassification, AutoTokenizer, TFAutoModelForSequenceClassification
import traceback
import threading
import re  # <--- CRITICAL IMPORT FOR LINK STRIPPING

# Import XAI Service
from backend.xai_service import get_explanation, stream_explanation

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
    print("Loading Amazon DeBERTa Model...")
    print("---------------------------------------------------------------")
    try:
        amazon_tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_DIR_AMAZON)
        amazon_model = TFDebertaV2ForSequenceClassification.from_pretrained(MODEL_DIR_AMAZON)
        print("Amazon model loaded successfully.")
    except Exception as e:
        print(f"Error loading Amazon model: {e}")

def get_social_model():
    global social_model, social_tokenizer
    if social_model is None:
        print(f"Loading Social TinyBERT model...")
        try:
            social_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR_SOCIAL)
            social_model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_DIR_SOCIAL)
            print("Social model loaded successfully.")
        except Exception as e:
            print(f"Error loading Social model: {e}")
            return None, None
    return social_model, social_tokenizer

# --- HELPER: TEXT CLEANING ---
def strip_links(text):
    """Removes http/https/www links to test content without url bias"""
    return re.sub(r'http\S+|www\.\S+', '', text).strip()

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
    
    # Return Label, Confidence, AND Raw Breakdown
    scores = {"GENUINE": real_score, "FAKE": fake_score}
    return label, min(conf, 0.99), scores

def process_social_probs(probs):
    bot_score = float(probs[1])
    human_score = float(probs[0])
    
    if bot_score > 0.70:
        label = "AI"
        conf = bot_score
    else:
        label = "HUMAN"
        conf = human_score
        
    # Return Label, Confidence, AND Raw Breakdown
    scores = {"HUMAN": human_score, "AI": bot_score}
    return label, min(conf, 0.99), scores

# --- ROUTES ---

# Batch prediction endpoint (Amazon)
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
            label, conf, scores = process_amazon_probs(probs)
            # Add 'scores' to output
            results.append({'label': label, 'confidence': conf, 'scores': scores})

        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Single review prediction endpoint (Amazon)
@app.route('/predict', methods=['POST'])
def predict_review():
    if not amazon_model: return jsonify({'error': 'Amazon Model is loading.'}), 503
    try:
        data = request.json
        text = data.get('text', '')
        
        inputs = amazon_tokenizer([text], return_tensors="tf", truncation=True, padding=True, max_length=128)
        logits = amazon_model(inputs).logits
        probs = tf.nn.softmax(logits, axis=1).numpy()[0]
        
        label, conf, scores = process_amazon_probs(probs)
        
        return jsonify({'label': label, 'confidence': conf, 'scores': scores})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Single comment prediction endpoint (Social) WITH SMART LOGIC
@app.route('/predict_comment', methods=['POST'])
def predict_comment():
    model, tokenizer = get_social_model()
    if not model: return jsonify({'error': 'Social Model missing'}), 500
    try:
        data = request.json
        text = data.get('text', '')
        
        # 1. Run Original Inference
        inputs = tokenizer([text], return_tensors="tf", truncation=True, padding=True, max_length=128)
        probs_orig = tf.nn.softmax(model(inputs).logits, axis=1).numpy()[0]
        
        # 2. Run Stripped (Link Check)
        text_stripped = strip_links(text)
        has_link = len(text) > len(text_stripped)
        
        label, conf, scores = process_social_probs(probs_orig)

        # 3. Smart Override Logic
        if has_link:
            inputs_strip = tokenizer([text_stripped], return_tensors="tf", truncation=True, padding=True, max_length=128)
            probs_strip = tf.nn.softmax(model(inputs_strip).logits, axis=1).numpy()[0]
            
            bot_score_orig = float(probs_orig[1])
            bot_score_strip = float(probs_strip[1])

            # Override if removing link flips AI -> Human
            if bot_score_orig > 0.70 and bot_score_strip < 0.60:
                print(f"[SmartFix] Override: Link caused FP. Orig: {bot_score_orig:.2f} -> Strip: {bot_score_strip:.2f}")
                # Use the STRIPPED probability for the final output
                label, conf, scores = process_social_probs(probs_strip)

        return jsonify({'label': label, 'confidence': conf, 'scores': scores})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Batch comment prediction endpoint (Social) WITH SMART LOGIC
@app.route('/predict_comment_batch', methods=['POST'])
def predict_comment_batch():
    model, tokenizer = get_social_model()
    if not model: return jsonify({'error': 'Social Model missing'}), 500
    try:
        data = request.json
        original_texts = data.get('texts', [])
        if not original_texts: return jsonify({'results': []})

        # 1. Prepare Stripped Versions
        stripped_texts = [strip_links(t) for t in original_texts]

        # 2. Run Inference
        inputs_orig = tokenizer(original_texts, return_tensors="tf", truncation=True, padding=True, max_length=128)
        logits_orig = model(inputs_orig).logits
        probs_orig = tf.nn.softmax(logits_orig, axis=1).numpy()

        inputs_strip = tokenizer(stripped_texts, return_tensors="tf", truncation=True, padding=True, max_length=128)
        logits_strip = model(inputs_strip).logits
        probs_strip = tf.nn.softmax(logits_strip, axis=1).numpy()
        
        results = []
        for i in range(len(original_texts)):
            # Default to original scores
            label, conf, scores = process_social_probs(probs_orig[i])
            
            # Smart Override Check
            bot_score_orig = float(probs_orig[i][1])
            bot_score_strip = float(probs_strip[i][1])
            has_link = len(original_texts[i]) > len(stripped_texts[i])

            if has_link and bot_score_orig > 0.70 and bot_score_strip < 0.60:
                 # Use the stripped scores
                 label, conf, scores = process_social_probs(probs_strip[i])
            
            results.append({'label': label, 'confidence': conf, 'scores': scores})
            
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# STREAMING EXPLANATION ENDPOINT
@app.route('/explain_stream', methods=['GET'])
def explain_stream():
    # Get params from query string (GET request is easier for EventSource/Streams)
    text = request.args.get('text', '')
    label = request.args.get('label', 'UNKNOWN')
    confidence = float(request.args.get('confidence', 0))

    def generate():
        # Yield chunks directly to the client
        for chunk in stream_explanation(text, label, confidence):
            if chunk:
                # We just send raw text for simple streaming fetch
                yield chunk

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