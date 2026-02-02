# Fake Review Detector

## Overview
The Fake Review Detector is a sophisticated machine learning application designed to detect and classify fraudulent reviews with high accuracy. Powered by state-of-the-art transformer models and explainable AI techniques, this project provides both programmatic API endpoints and a seamless browser extension for real-time review authentication.

## Features
- **Dual-Model Architecture**: DeBERTa model for Amazon review classification and TinyBERT model for social media/AI-generated content detection
- **Explainable AI (XAI)**: Provides detailed explanations and feature importance for predictions
- **Batch Processing API**: Efficient processing of multiple reviews simultaneously
- **Lazy Loading**: Optimized model initialization for improved performance
- **REST API**: Comprehensive Flask-based API with CORS enabled for cross-origin requests
- **Browser Extension**: Chrome extension for seamless in-browser review authentication
- **Confidence Scoring**: Capped confidence metrics with nuanced prediction categories (Fake, Genuine, Uncertain, AI, Human)

## Project Structure
```
├── app.py                       # Flask application with dual model inference
├── README.md                    # Project documentation
├── requirements.txt             # Python package dependencies
├── backend/                     # Backend services and models
│   ├── __init__.py             # Package initialization
│   ├── xai_service.py          # Explainable AI service for model interpretability
│   ├── model/                  # Amazon Review DeBERTa Model
│   │   ├── added_tokens.json   # Tokenizer data
│   │   ├── config.json         # Model configuration
│   │   ├── special_tokens_map.json # Special tokens mapping
│   │   ├── spm.model           # SentencePiece model
│   │   ├── tf_model.h5         # Trained TensorFlow DeBERTa model
│   │   └── tokenizer_config.json # Tokenizer configuration
│   └── model_tinybert/         # Social Media TinyBERT Model
│       ├── config.json         # Model configuration
│       ├── special_tokens_map.json # Special tokens mapping
│       ├── tf_model.h5         # Trained TinyBERT model
│       ├── tokenizer_config.json # Tokenizer configuration
│       └── vocab.txt           # Model vocabulary
├── data/                        # Training and evaluation datasets
│   ├── fake_reviews_dataset.csv # Amazon reviews dataset
│   └── hc3_flattened_balanced.csv # Social media/AI content dataset
├── notebooks/                   # Jupyter notebooks for research and development
│   ├── transformer.ipynb       # Model  Experimental variants
│   └── botorwot.ipynb          # Bot/Human classification notebook
└── extension/                   # Chrome browser extension
    ├── background.js           # Extension background script
    ├── content.js              # Content script for DOM manipulation
    ├── manifest.json           # Extension configuration manifest
    ├── icons/                  # Extension icon assets
    └── popup/                  # Extension popup interface
        ├── popup.css           # Popup styling
        ├── popup.html          # Popup markup
        └── popup.js            # Popup functionality
```

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Local Development
1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd fakeReviewDetector
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # macOS/Linux
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask application**:
   ```bash
   python app.py
   ```
   The API server will be available at `http://localhost:5000`

### Browser Extension Setup
1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode" (toggle in top right corner)
3. Click "Load unpacked"
4. Select the `extension` folder from this project
5. The extension will appear in your Chrome toolbar

## API Endpoints

### Batch Prediction
```
POST /predict_batch
Content-Type: application/json

{
  "texts": ["review text 1", "review text 2", ...]
}

Response:
{
  "results": [
    {
      "label": "GENUINE|FAKE|UNCERTAIN|AI|HUMAN",
      "confidence": 0.95
    },
    ...
  ]
}
```

### Single Review Classification
```
POST /predict
Content-Type: application/json

{
  "text": "review text",
  "model_type": "amazon|social"
}

Response:
{
  "label": "GENUINE|FAKE|UNCERTAIN|AI|HUMAN",
  "confidence": 0.95,
  "explanation": {...}
}
```

## Usage

### Python API
```python
import requests

response = requests.post(
    'http://localhost:5000/predict_batch',
    json={'texts': ['Your review text here']}
)
predictions = response.json()
```

### Browser Extension
Once installed, the extension automatically analyzes reviews as you browse. Results are displayed directly in the extension popup with confidence scores and explanations.

## Model Details

### Amazon Reviews (DeBERTa)
- **Architecture**: DeBERTa V2
- **Training Data**: Curated Amazon reviews dataset
- **Output Classes**: GENUINE, FAKE, UNCERTAIN
- **Optimization**: CPU-based inference for broader compatibility
### Social Media / AI-Generated Content (TinyBERT)
- **Architecture**: TinyBERT (lightweight BERT variant)
- **Training Data**: HC3 balanced dataset
- **Output Classes**: HUMAN, AI
- **Optimization**: Lazy loading on demand for resource efficiency

## Technical Stack
- **Deep Learning Framework**: TensorFlow 2.10+
- **NLP Library**: Hugging Face Transformers
- **Web Framework**: Flask with CORS support
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Tokenization**: SentencePiece, Transformers Tokenizers

## Performance Considerations
- Models are cached in memory after first load for optimal performance
- Batch processing API supports high-throughput inference
- CPU-based inference ensures broad compatibility (GPU support available)
- Confidence scores are capped at 99% for realistic uncertainty representation

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [Hugging Face](https://huggingface.co/) - Transformers library
- [TensorFlow](https://www.tensorflow.org/) - Deep learning framework
- [Flask](https://flask.palletsprojects.com/) - Web framework
- Dataset contributors and research community
