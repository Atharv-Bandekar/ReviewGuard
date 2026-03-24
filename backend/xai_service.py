import torch
import threading
import gc
from functools import lru_cache
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

# ---------------- CONFIG ----------------
LOCAL_LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
# Sweet spot for token length: Enough for a solid paragraph, prevents cut-offs
MAX_NEW_TOKENS = 150   
TEMPERATURE = 0.6     

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
                if _llm_tokenizer.pad_token is None:
                    _llm_tokenizer.pad_token = _llm_tokenizer.eos_token
                    
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
    if "Genuine-style, Human-written" in label:
        return "This reads naturally. The varied sentence structure and specific details strongly suggest a real person wrote this."
    elif "Genuine-style, AI-assisted" in label:
        return "While the feedback is balanced, the hyper-perfect grammar and uniform pacing indicate an AI likely helped draft this."
    elif "Promotional-style, Human-written" in label:
        return "This is clearly human-written, but the aggressive marketing language and extreme enthusiasm suggest it might be biased or incentivized."
    elif "Promotional-style, AI-assisted" in label:
        return "This has all the hallmarks of a bot. It relies heavily on generic buzzwords, repetitive phrasing, and an unnatural promotional tone."
    return "Analysis unavailable."

# ---------------- PROMPT BUILDER (THE SWEET SPOT) ----------------

def build_prompt(text, label, confidence):
    # 1. System Prompt: The Sweet Spot Constraints
    system_prompt = (
        "You are an expert linguistic analyst. Your job is to justify why a review received a specific classification. "
        "STRICT RULES:\n"
        "1. Write EXACTLY 2 to 3 fluid, conversational sentences (around 50-60 words total).\n"
        "2. Do NOT use lists, bullet points, or markdown formatting.\n"
        "3. Justify the classification by pointing out specific words, phrasing, or structural patterns from the text.\n"
        "4. Speak directly to the user in a natural tone (e.g., 'This reads naturally because...', 'Notice how this uses...').\n"
        "5. Never use robotic phrasing like 'The system classified' or 'The model detected'."
    )
    
    # 2. Dynamic Focus: Telling it exactly how to justify the label
    if "Genuine-style, Human-written" in label:
        focus = "Justify this as human by pointing out specific natural details, varied sentence pacing, or authentic emotion."
    elif "Genuine-style, AI-assisted" in label:
        focus = "Justify this as AI-assisted by pointing out overly perfect grammar, uniform sentence lengths, or AI-like vocabulary despite the helpful tone."
    elif "Promotional-style, Human-written" in label:
        focus = "Justify this as promotional by pointing out the specific aggressive marketing language, high bias, or exaggerated enthusiasm."
    elif "Promotional-style, AI-assisted" in label:
         focus = "Justify this as an AI bot by pointing out the specific robotic buzzwords, repetitive structure, and unnatural marketing tone."
    else:
         focus = "Briefly justify the tone and structure of the text."

    # 3. Delimiter-Based User Prompt
    user_prompt = f"""<CONTEXT>
Classification: {label}
Goal: {focus}
</CONTEXT>

<REVIEW_TEXT>
{text[:300]}
</REVIEW_TEXT>

<INSTRUCTION>
Write your 2-3 sentence justification now.
</INSTRUCTION>"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    if _llm_tokenizer and getattr(_llm_tokenizer, "chat_template", None):
        return _llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    return f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"

# ---------------- STREAMING GENERATOR ----------------

def stream_explanation(text, label, confidence):
    """Yields text chunks as they are generated."""
    load_local_llm()
    if _llm_model == "FAILED" or _llm_tokenizer is None:
        yield analyze_locally(text, label) 
        return

    try:
        prompt = build_prompt(text, label, confidence)
        
        inputs = _llm_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        streamer = TextIteratorStreamer(_llm_tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = dict(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            streamer=streamer,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=_llm_tokenizer.pad_token_id
        )

        thread = threading.Thread(target=_llm_model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text
            
        gc.collect()

    except Exception as e:
        yield f"Error: {str(e)}"

# ---------------- STATIC GENERATOR (LEGACY SUPPORT) ----------------

@lru_cache(maxsize=100)
def get_explanation(text, label, confidence):
    full_text = ""
    for chunk in stream_explanation(text, label, confidence):
        full_text += chunk
    return full_text