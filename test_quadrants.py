import requests
import time
import json

API_URL = "http://127.0.0.1:8000/predict"

# --- TEST CASES DESIGNED FOR REVIEWGUARD HEURISTICS ---
test_cases = [
    {
        "id": "Q1",
        "expected_quadrant": "Genuine-style, Human-written",
        "description": "Casual phrasing, personal pronouns ('I', 'my'), mixed sentiment.",
        "text": "I bought this coffee maker about three months ago. It works okay for morning brews, but the lid is really hard to clean out. I dropped the carafe once and it didn't break, which is a huge plus. Overall, not a bad machine for the price, but I probably wouldn't buy this exact model again."
    },
    {
        "id": "Q2",
        "expected_quadrant": "Genuine-style, AI-assisted",
        "description": "Zero personal pronouns, complex vocabulary (>7 chars), uniform sentence pacing.",
        "text": "The apparatus provides adequate functionality for daily utilization. However, the ergonomic configuration exhibits notable deficiencies during prolonged operation. Consequently, while the operational metrics remain satisfactory, the physical construction requires substantial enhancement to ensure optimal consumer satisfaction."
    },
    {
        "id": "Q3",
        "expected_quadrant": "Promotional-style, Human-written",
        "description": "Hype formatting, all caps, multiple exclamations, but retains human pacing/slang.",
        "text": "They remain that perfect touch to me with that thing I wish they held a little more space."
    },
    {
        "id": "Q4",
        "expected_quadrant": "Promotional-style, AI-assisted",
        "description": "Marketing template, robotic transitions, zero pronouns, highly sterile.",
        "text": "Complements different color schemes, that are more vibrant plus more comfortable. That functionality exceeds expectations, with intuitive controls with responsive performance. This materials science involved must be advanced, given the performance characteristics observed."
    }
]

def run_tests():
    print("======================================================")
    print("🚀 Initiating ReviewGuard Quadrant Evaluation")
    print("======================================================")
    
    passed_tests = 0
    
    for idx, case in enumerate(test_cases, 1):
        print(f"\n[Test {idx}/4] Targeting: {case['expected_quadrant']}")
        print(f"Logic: {case['description']}")
        
        payload = {"text": case["text"]}
        
        try:
            start_time = time.time()
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()
            elapsed_time = time.time() - start_time
            
            result = response.json()
            returned_label = result.get("label", "UNKNOWN")
            confidence = result.get("confidence", 0)
            scores = result.get("scores", {})
            
            print(f"Result: {returned_label} ({confidence * 100:.1f}%)")
            print(f"Scores -> Style (Fraud): {scores.get('fraud_style_score', 0):.2f} | Authorship (AI): {scores.get('ai_likelihood_score', 0):.2f}")
            print(f"Latency: {elapsed_time:.2f}s")
            
            # Note: We are doing a soft match here because the exact label might 
            # differ slightly depending on your confidence thresholds (e.g., "Uncertain-style")
            if case["expected_quadrant"].split(",")[0] in returned_label and case["expected_quadrant"].split(",")[1] in returned_label:
                print("✅ PASS")
                passed_tests += 1
            else:
                print("❌ FAIL (Label mismatch)")
                
        except requests.exceptions.ConnectionError:
            print("❌ FAIL: Could not connect to API. Is app.py running?")
            return
        except Exception as e:
            print(f"❌ FAIL: API Error -> {str(e)}")

    print("\n======================================================")
    print(f"🏁 Testing Complete: {passed_tests}/4 Passed")
    
    if passed_tests < 4:
        print("\nNote: If AI-assisted tests are failing, check line 58 in app.py.")
        print("The author axis is currently hardcoded to 'Human-written'.")
    print("======================================================")

if __name__ == "__main__":
    run_tests()