import torch
import threading
import gc
from functools import lru_cache
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

# ---------------- CONFIG ----------------
LOCAL_LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 100 
TEMPERATURE = 0.3     

# ---------------- LOAD MODEL ----------------
_llm_tokenizer = None
_llm_model = None
_model_lock = threading.Lock()

def load_local_llm():
    global _llm_model, _llm_tokenizer
    if _llm_model is not None: return

    with _model_lock:
        if _llm_model is None:
            print(f"[XAI] ⏳ Loading local LLM ({LOCAL_LLM_MODEL})...")
            try:
                _llm_tokenizer = AutoTokenizer.from_pretrained(LOCAL_LLM_MODEL)
                _llm_model = AutoModelForCausalLM.from_pretrained(
                    LOCAL_LLM_MODEL,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    low_cpu_mem_usage=True
                )
                _llm_model.eval()
                print("[XAI] ✅ Local LLM loaded successfully.")
            except Exception as e:
                print(f"[XAI] ❌ Failed to load Local LLM: {e}")
                _llm_model = "FAILED"

# Background load
threading.Thread(target=load_local_llm, daemon=True).start()

# ---------------- FALLBACK LOGIC ----------------

def analyze_locally(text, label):
    """Fallback rule-based logic when AI fails."""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    has_link = "http" in text or "www" in text
    
    if label == "FAKE":
        return "The review lacks specific details and uses generic, overly enthusiastic language."
    elif label == "GENUINE":
        return "The review contains balanced feedback and specific usage scenarios."
    elif label in ["BOT", "AI"]:
        if has_link: return "Contains external links, common in automated spam."
        return "Follows repetitive patterns typical of automated scripts."
    elif label == "HUMAN":
        if has_link: return "Contains a link, but context and tone suggest a human author."
        return "Shows emotional nuance and natural conversational phrasing."
    return "Analysis unavailable."

# ---------------- PROMPT BUILDER ----------------

def build_prompt(text, label, confidence):
    prompt = ""
    
    if label in ["BOT", "AI"]:
        prompt = (
            f"The system classified this comment as AI/BOT with {confidence:.0%} confidence. "
            "Explain the decision by describing which textual or structural patterns influenced the model. "
            "Do not assume intent. Mention links or generic phrasing if relevant. "
            "Limit to 40 words."
        )
    elif label == "HUMAN":
        prompt = (
            f"The system classified this comment as HUMAN with {confidence:.0%} confidence. "
            "Explain which language patterns aligned with typical human-written content. "
            "Limit to 40 words."
        )
    elif label == "FAKE":
        prompt = (
            f"The system classified this review as FAKE with {confidence:.0%} confidence. "
            "Explain which linguistic patterns aligned with fake reviews (e.g., generic buzzwords). "
            "Limit to 40 words."
        )
    else: # GENUINE
        prompt = (
            f"The system classified this review as GENUINE with {confidence:.0%} confidence. "
            "Explain which language features aligned with genuine reviews (e.g., specific details). "
            "Limit to 40 words."
        )

    if confidence < 0.80:
        prompt += " Acknowledge that signals were mixed."

    messages = [
        {"role": "system", "content": "You are an analytical AI tool explaining model behavior. Be precise and objective."},
        {"role": "user", "content": f"{prompt}\n\nAnalysis Text: \"{text[:300]}\""}
    ]
    
    if _llm_tokenizer.chat_template:
        return _llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"System: Analyze.\nUser: {prompt}\nText: {text[:300]}\nAssistant:"

# ---------------- STREAMING GENERATOR ----------------

def stream_explanation(text, label, confidence):
    """Yields text chunks as they are generated."""
    load_local_llm()
    if _llm_model == "FAILED":
        yield analyze_locally(text, label) # Fallback to textblob if LLM dies
        return

    try:
        prompt = build_prompt(text, label, confidence)
        inputs = _llm_tokenizer(prompt, return_tensors="pt")
        
        # 1. REMOVE any print statements here
        # print(f"[XAI] Generating...") <--- DELETE THIS

        # 2. Setup the Iterator Streamer (This captures text, doesn't print it)
        streamer = TextIteratorStreamer(_llm_tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=_llm_tokenizer.eos_token_id
        )

        # 3. Start generation in a thread
        thread = threading.Thread(target=_llm_model.generate, kwargs=generation_kwargs)
        thread.start()

        # 4. Yield tokens (The terminal remains silent)
        for new_text in streamer:
            yield new_text
            
        gc.collect()

    except Exception as e:
        yield f"Error: {str(e)}"

# ---------------- STATIC GENERATOR (LEGACY SUPPORT) ----------------

@lru_cache(maxsize=100)
def get_explanation(text, label, confidence):
    """Non-streaming version for the old /explain endpoint."""
    # We can just consume the stream to build the string
    full_text = ""
    for chunk in stream_explanation(text, label, confidence):
        full_text += chunk
    return full_text