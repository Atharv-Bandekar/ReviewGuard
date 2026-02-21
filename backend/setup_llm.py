import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# Define the model to download
LOCAL_LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

def download_model():
    print(f"⬇️  Downloading {LOCAL_LLM_MODEL}...")
    
    # This downloads and caches the model tokenizer and weights locally
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(LOCAL_LLM_MODEL)
    
    print(f"✅ Model downloaded successfully to Hugging Face cache.")
    print(f"   You can now run 'python app.py' and it will load instantly.")

if __name__ == "__main__":
    download_model()