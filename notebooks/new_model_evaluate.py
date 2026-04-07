import os
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import DebertaV2Tokenizer, TFDebertaV2ForSequenceClassification
from tqdm import tqdm

# 1. Configuration
CSV_PATH = "live_shadow_run.csv"  
OUTPUT_PATH = "new_model_comparison.csv"

# FIX: Step up one level from 'notebooks' to the project root
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE, "backend", "model")

MAX_LEN = 128
BATCH_SIZE = 32

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Force CPU (or remove if using GPU)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Reduce TF spam

print("\n=== STARTING BATCH EVALUATION ===")

# 2. Load the Dataset
if not os.path.exists(CSV_PATH):
    print(f"Error: Could not find {CSV_PATH}. Please check the filename.")
    exit()

df = pd.read_csv(CSV_PATH)
# Ensure we have the review text and handle any blank rows
df = df.dropna(subset=['review_text'])
texts = df['review_text'].astype(str).tolist()
print(f"Loaded {len(texts)} reviews from {CSV_PATH}")

# 3. Load the Current Model
print("Loading DeBERTa Tokenizer and Model...")
try:
    tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_DIR)
    model = TFDebertaV2ForSequenceClassification.from_pretrained(MODEL_DIR)
except Exception as e:
    print(f"Failed to load model: {e}")
    exit()

# 4. Batch Inference Loop
new_fake_probs = []
new_real_probs = []

print("Running inference with current model weights...")
for i in tqdm(range(0, len(texts), BATCH_SIZE)):
    batch_texts = texts[i:i+BATCH_SIZE]
    
    # Tokenize the batch
    inputs = tokenizer(
        batch_texts, 
        return_tensors="tf", 
        truncation=True, 
        padding=True, 
        max_length=MAX_LEN
    )
    
    # Get predictions
    logits = model(inputs).logits
    probs = tf.nn.softmax(logits, axis=1).numpy()
    
    # Extract raw probabilities (Index 0 = Genuine, Index 1 = Promotional/Fake)
    for prob in probs:
        new_real_probs.append(float(prob[0]))
        new_fake_probs.append(float(prob[1]))

# 5. Save the Results
# Append the new scores alongside the old ones for easy comparison
df['new_fake_prob'] = new_fake_probs
df['new_real_prob'] = new_real_probs

# Calculate the difference so you can instantly see the shift in paranoia
if 'fake_prob' in df.columns:
    df['prob_difference'] = df['new_fake_prob'] - df['fake_prob']

df.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Evaluation complete! Results saved to '{OUTPUT_PATH}'")