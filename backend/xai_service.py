import requests
import json
import os
import random
from functools import lru_cache
from dotenv import load_dotenv
from textblob import TextBlob

load_dotenv()

# 🔴 CONFIGURATION
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# 🚀 FREE MODEL LIST
FREE_MODELS = [
    "meta-llama/llama-3.2-3b-instruct:free",      
    "google/gemini-2.0-flash-exp:free",           
    "google/gemma-3-12b-it:free",                 
    "mistralai/mistral-small-3.1-24b-instruct:free" 
]

@lru_cache(maxsize=100)
def get_cached_explanation(review_snippet, label, confidence):
    return generate_explanation_with_fallback(review_snippet, label, confidence)

# 🧠 SMART LOCAL ENGINE (Fallback)
def analyze_locally(text, label):
    """
    Rule-based logic when AI fails. 
    Now explicitly handles 'HUMAN' and 'AI' (from app.py social route)
    alongside 'GENUINE' and 'FAKE' (from amazon route).
    """
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity       # -1.0 to 1.0
    subjectivity = blob.sentiment.subjectivity # 0.0 to 1.0
    word_count = len(text.split())

    # --- 1. AMAZON REVIEWS (Fake/Genuine) ---
    if label == "FAKE":
        if polarity > 0.8: return "Suspiciously over-enthusiastic and generic."
        if word_count < 10: return "Too short and vague to be verified."
        return "Lacks specific usage details typical of a genuine owner."
    
    elif label == "GENUINE":
        if word_count > 30: return "Contains specific details and balanced feedback."
        return "Writing style is natural and consistent with a real user."

    # --- 2. SOCIAL MEDIA COMMENTS (AI/Human) ---
    # Note: app.py returns "AI" for bots, but we check "BOT" just in case.
    elif label in ["AI", "BOT"]:
        if "http" in text or "www" in text:
            return "Contains external links, a common trait of spam bots."
        elif "check my" in text.lower() or "subscribe" in text.lower():
            return "Promotional language typical of self-promotion bots."
        elif word_count < 5:
            return "Extremely short and generic interaction."
        elif polarity > 0.9:
            return "Generic, excessive praise often used to boost engagement."
        else:
            return "Follows repetitive patterns typical of automated scripts."

    elif label == "HUMAN":
        if subjectivity > 0.5:
            return "Shows personal opinion and emotional nuance."
        elif "?" in text:
            return "Asks a relevant context-aware question."
        elif word_count > 15:
            return "Sentence structure is complex and conversational."
        else:
            return "Natural, conversational phrasing."

    return "Analysis unavailable."

def generate_explanation_with_fallback(text, label, confidence):
    # 1. Select Prompt based on Label Type
    prompt = ""
    
    # CASE A: Social Bot/AI
    if label in ["BOT", "AI"]:
        prompt = (
            f"This social media comment is flagged as AI/BOT ({confidence:.0%} certainty). "
            "In 1 short sentence, explain why. "
            "Look for signs like: scam links, irrelevant self-promotion, robotic repetition, or context-less praise. "
            "Speak naturally."
        )
    
    # CASE B: Social Human
    elif label == "HUMAN":
        prompt = (
            f"This comment looks like a real HUMAN ({confidence:.0%} certainty). "
            "In 1 short sentence, explain why. "
            "Mention signs like: specific reaction to the content, slang, typos, or emotional nuance. "
            "Speak naturally."
        )

    # CASE C: Amazon Fake
    elif label == "FAKE":
        prompt = (
            f"This product review is flagged as FAKE ({confidence:.0%} certainty). "
            "In 1 short sentence, explain why. "
            "Look for: generic marketing buzzwords, lack of specific details, or robotic enthusiasm. "
            "Speak naturally."
        )

    # CASE D: Amazon Genuine
    else: # GENUINE
        prompt = (
            f"This product review looks GENUINE ({confidence:.0%} certainty). "
            "In 1 short sentence, explain why. "
            "Mention how it offers balanced pros/cons or specific usage scenarios. "
            "Speak naturally."
        )

    # 2. Try Models (With reduced timeout)
    # Only try AI if we have a key, otherwise jump to local
    if OPENROUTER_API_KEY:
        for model in FREE_MODELS:
            try:
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": f"{prompt}\n\nTEXT:\n{text[:300]}"}],
                    "temperature": 0.7, 
                }
                
                headers = {
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost:8000", 
                    "X-Title": "ReviewGuard"
                }

                # ⚡ FAST TIMEOUT: 4 Seconds ⚡
                response = requests.post(OPENROUTER_URL, headers=headers, data=json.dumps(payload), timeout=8)
                
                if response.status_code == 200:
                    content = response.json()['choices'][0]['message']['content'].strip()
                    # Cleanup: remove quotes if the model adds them
                    content = content.replace('"', '').replace("'", "")
                    if content: return content
                
            except Exception:
                continue # Try next model or fall through
    else:
        print("⚠️ No API Key found. Skipping AI models.")

    # 3. Fallback
    print(f"⚠️ AI unavailable for label '{label}'. Using Local Logic.")
    return analyze_locally(text, label)

def get_explanation(review_text, label, confidence):
    # Cache key is short text + label to avoid long keys
    return get_cached_explanation(review_text[:100], label, confidence)