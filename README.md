
# ReviewGuard: E-Commerce Truth Detector

## Overview
ReviewGuard is a highly specialized machine learning application designed to detect and classify fraudulent product reviews with high accuracy. Moving beyond simple "Fake vs. Real" binaries, this system decomposes review analysis into two orthogonal dimensions: **Traditional Fraud-Style Detection** and **AI-Authorship Likelihood**. 

Powered by a local DeBERTa-v3 Small sequence classifier, an algorithmic Heuristics Engine, and the Groq API (LLaMA 3.1) for lightning-fast Explainable AI (XAI), ReviewGuard provides both REST API endpoints and a seamless Chrome browser extension for real-time authentication on e-commerce platforms like Amazon.

## Core Features
- **Two-Axis Classification**: Classifies reviews into 4 nuanced quadrants:
  - ✅ Genuine-style, Human-written
  - ⚠️ Genuine-style, AI-assisted
  - 🚫 Promotional-style, Human-written
  - 🚫 Promotional-style, AI-assisted
- **Explainable AI (XAI) Streaming**: Uses the Groq API (`llama-3.1-8b-instant`) to generate ultra-crisp, conversational explanations of *why* a review was flagged, streamed dynamically to the UI. Includes a pure-Python templated fallback.
- **Algorithmic Heuristics**: Calculates AI generation probability dynamically using Lexical Complexity and Sentence Length Uniformity, cross-referenced with a temperature-calibrated probability matrix.
- **Batch Processing API**: Efficient processing of multiple reviews simultaneously with confidence scores mathematically scaled.
- **Browser Extension**: Specialized Chrome extension that injects real-time confidence badges and XAI "Why?" buttons directly into Amazon product pages.

## Project Structure
```text
├── app.py                       # Core Flask application with dual-axis inference
├── requirements.txt             # Python dependencies (TensorFlow, Transformers, etc.)
├── .env                         # Environment variables (Groq API Key)
├── README.md                    # Project documentation
├── backend/                     # Backend services and models
│   ├── __init__.py              
│   ├── xai_service.py           # Groq API streaming service for Explainable AI
│   ├── heuristics_engine.py     # Rule-based authorship analysis
│   └── model/                   # Local DeBERTa-v3 Small weights
│       ├── config.json          
│       ├── tf_model.h5          
│       └── tokenizer_config.json 
├── notebooks/                   # Jupyter notebooks for research and development
└── extension/                   # Chrome browser extension
    ├── content.js               # Content script for DOM manipulation on Amazon
    ├── manifest.json            # Extension configuration manifest (Manifest V3)
    ├── icons/                   # Extension icon assets
    └── popup/                   # Extension popup interface

```

## Installation & Setup

### Prerequisites

  - Python 3.8 or higher
  - `pip` package manager
  - Git
  - Groq API Key

### Local Backend Setup

1.  **Clone the repository**:

    ```bash
    git clone https://github.com/Atharv-Bandekar/fakeReviewDetector.git
    cd fakeReviewDetector
    ```

2.  **Create and activate a virtual environment**:

    ```bash
    # Windows
    python -m venv venv
    venv\Scripts\activate      

    # macOS/Linux
    python3 -m venv venv
    source venv/bin/activate   
    ```

3.  **Install dependencies**:
    Installs required ML frameworks (`tensorflow`, `torch`) and NLP libraries (`transformers>=4.40.0`).

    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables**:
    Create a .env file in the root directory and add your Groq API key:

    ```bash
    GROQ_API_KEY=your_groq_api_key_here
    ```

5.  **Start the API Server**:

    ```bash
    python app.py
    ```

    *The server will initialize on `http://127.0.0.1:8000`.*

### Browser Extension Setup

1.  Open Google Chrome and navigate to `chrome://extensions/`
2.  Enable **Developer mode** (toggle in the top right corner).
3.  Click **Load unpacked**.
4.  Select the `extension` folder located inside this project directory.
5.  Navigate to any Amazon product page to see the floating "Scan Page" widget.

## API Endpoints

### 1\. Batch Prediction

Analyzes multiple reviews simultaneously and returns the 4-quadrant label.

```http
POST /predict_batch
Content-Type: application/json

{
  "texts": ["This product is amazing!", "Bought this after 5 months of research."]
}
```

**Response:**

```json
{
  "results": [
    {
      "label": "Genuine-style, Human-written",
      "confidence": 0.95,
      "scores": {
        "fraud_style_score": 0.05,
        "genuine_style_score": 0.95,
        "ai_likelihood_score": 0.12
      }
    }
  ]
}
```

### 2\. Explainable AI (Streaming)

Generates a 2-3 sentence conversational justification for the model's prediction.

```http
GET /explain_stream?text={url_encoded_review_text}&label={label}&confidence={confidence}
```

**Response:**
Returns a chunked `text/plain` stream (Server-Sent Events format compatible) yielding tokens in real-time.

## System Architecture Details

### 1\. Fraud-Style Axis (DeBERTa)

  - **Architecture**: DeBERTa V2 (`TFDebertaV2ForSequenceClassification`)
  - **Purpose**: Detects traditional incentivized, biased, or highly promotional language patterns.
  - **Optimization**: Forced CPU-based inference (`CUDA_VISIBLE_DEVICES="-1"`) for broader deployment compatibility.

### 2\. Authorship Axis (Heuristics Engine)

  - **Architecture**: Dynamic Linguistic Algorithm (O(N) time complexity).
  - **Purpose**: Calculates the mathematical likelihood of AI assistance by analyzing **Lexical Complexity** (polysyllabic density / Type-Token Ratio) and **Sentence Length Uniformity** (variance analysis).

### 3\. Explainable AI (Qwen LLM)

Generates a direct, analytical justification for the model's prediction.

```
POST /explain_stream
Content-Type: application/json

{
  "text": "The URL encoded review text...",
  "label": "Genuine-style, Human-written",
  "confidence": 0.95
}

```

Response:
Returns a chunked text/plain stream yielding tokens in real-time.

## System Architecture Details
1. **Fraud-Style Axis (DeBERTa)**
- Architecture: DeBERTa-v3 Small (TFDebertaV2ForSequenceClassification)
- Purpose: Detects traditional incentivized, biased, or highly promotional language patterns.
- Optimization: Forced CPU-based inference (CUDA_VISIBLE_DEVICES="-1") for broader deployment compatibility with temperature-calibrated softmax outputs.

2. **Authorship Axis (Heuristics Engine)**
- Architecture: Dynamic Linguistic Algorithm (O(N) time complexity).
- Purpose: Calculates the mathematical likelihood of AI assistance by analyzing Lexical Complexity (polysyllabic density / Type-Token Ratio) and Sentence Length Uniformity (variance analysis).

3. **Explainable AI (Groq API)**
- Architecture: llama-3.1-8b-instant via Groq
- Purpose: Translates the numeric outputs of the Two-Axis system into natural language justifications for end-users based on a strict cross-referenced prompt matrix.
- Optimization: API-driven to eliminate local memory overhead, providing ultra-low latency streaming with a built-in Python string template fallback to ensure continuous uptime.

## Demo

<img src="demo_reviewGuard.gif" width="80%" />

## Contributing

Contributions to improve detection algorithms or expand platform support are welcome.

1.  Fork the repository.
2.  Create a feature branch (`git checkout -b feature/NewHeuristic`).
3.  Commit your changes (`git commit -m 'Add new lexical density check'`).
4.  Push to the branch (`git push origin feature/NewHeuristic`).
5.  Open a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

