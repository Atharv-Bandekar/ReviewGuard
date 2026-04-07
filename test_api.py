import requests
import json

# The URL of your local Flask API
url = "http://127.0.0.1:8000/predict_batch"

# The 4 test cases we designed earlier
test_cases = [
    # Test 1: Genuine-style, Human-written
    "I bought these yoga sling shoes about three months ago for my retail job where I stand all day. Honestly, they are amazingly comfortable! It took a few days to get used to the weird matte bottom, but my lower back pain is completely gone now. The only downside is the funky tan lines I get if I wear them outside at the park. Highly recommend if you need good arch support!",
    
    # Test 2: Genuine-style, AI-assisted
    "The ergonomic construction of this footwear provides substantial amelioration for persistent musculoskeletal discomfort. Initial utilization necessitated a brief acclimation period regarding the unconventional footbed texture. Subsequent application throughout extended professional shifts demonstrated exceptional structural support and pressure distribution. Furthermore, the breathable material facilitates adequate ventilation during elevated ambient temperatures. The singular disadvantage involves unconventional pigmentation disparities upon prolonged ultraviolet exposure.",
    
    # Test 3: Promotional-style, Human-written
    "WOW BEST SHOES EVER!!! I am absolutely OBSESSED with this product it is the greatest thing I have ever bought in my entire life! You guys seriously need to buy this right now before it sells out! I give it 100 stars out of 10! Flawless perfect amazing beautiful! I will be buying ten more pairs for my entire family!",
    
    # Test 4: Promotional-style, AI-assisted
    "This multifaceted footwear represents a comprehensive game-changer within the contemporary consumer landscape. Moreover, its seamless integration and intricate design serve as a testament to unparalleled manufacturing excellence. We must delve into the crucial features that elevate this phenomenal product above generic alternatives. Utilizing this beacon of innovation will unequivocally transform your daily operational parameters. Purchasing this premium merchandise guarantees maximum satisfaction and ultimate lifestyle enhancement."
]

payload = {"texts": test_cases}
headers = {"Content-Type": "application/json"}

try:
    print("🚀 Sending batch request to ReviewGuard API...\n")
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    
    results = response.json().get('results', [])
    
    for i, res in enumerate(results):
        print(f"--- TEST CASE {i+1} ---")
        print(f"Label:      {res['label']}")
        print(f"Confidence: {res['confidence']*100:.1f}%")
        print(f"Breakdown:  Style: {res['scores']['fraud_style_score']:.2f} | AI Likelihood: {res['scores']['ai_likelihood_score']:.2f}\n")

except Exception as e:
    print(f"❌ Error connecting to API: {e}")