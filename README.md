# 🐍 Python Q&A Chatbot (Custom Word2Vec — LLM-Free)

A compact, explainable Python Q&A chatbot using a custom-trained Word2Vec model — no pre-trained language models or LLMs used. Designed for teaching, evaluation, and demonstration.

## What’s changed (current)
- Model files: `word2vec_custom_v2.model` and `word2vec_chatbot_model_v2.pkl`
- Trained on: ~83K Python Q&A pairs (project-specific dataset)
- Vocabulary: ~57K unique tokens
- Vector size: 300 dimensions
- Training stable mode: use `workers=1` to avoid BLAS/dot_float issues on some environments

## Highlights
- ✅ Custom Word2Vec trained from scratch (no pre-trained embeddings)
- ✅ Smart input validation (greeting detection, gibberish rejection)
- ✅ Streamlit interface for quick demos
- ✅ Evaluation tools included (top-k retrieval, cross-validation, real-world tests)

## Quick Start

1. Prerequisites
```bash
Python 3.8+  # or 3.10/3.11
pip
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Download NLTK data (one-time)
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

4. Place model files in the project root (after training or download from Colab):
- `word2vec_custom_v2.model`
- `word2vec_chatbot_model_v2.pkl`

5. Run the web app
```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Project layout (short)
```
.
├── app.py
├── word2vec_chatbot.py
├── question_validator.py
├── Build_word2vec_Colab.ipynb
├── word2vec_custom_v2.model
├── word2vec_chatbot_model_v2.pkl
├── requirements.txt
└── data/
        ├── conversational_dataset.csv
        ├── contextual_followups.csv
        └── learning_guidance.csv
```

## How it works (brief)
- Training: custom Word2Vec (skip-gram or CBOW depending on notebook config) on your Q&A corpus.
- Document vectors: average of in-vocabulary word vectors per text.
- Retrieval: cosine similarity between user query vector and stored question vectors; top-K candidates returned.
- Smart rules: short, exact conversational patterns (greetings, "python" clarification) are handled before retrieval.

## Evaluation & Sanity checks
- Built-in evaluation cell in `Build_word2vec_Colab.ipynb` provides:
    - Top-K retrieval accuracy (on held-out samples)
    - Similarity score distribution
    - Cross-validation-style retrieval metrics
    - A "Real-World" unseen-questions test to check generalization

Tip: 100% retrieval on training data is expected for a retrieval system (it finds the same example). Use the real-world unseen test to measure true generalization.

## Configuration tips
- If you see Gensim/BLAS `dot_float` errors on Colab or local machines, set `workers=1` in training.
- Adjust the confidence threshold and Top-K in `app.py` for stricter/looser answers.

## Troubleshooting
- Model files not found: place `word2vec_custom_v2.model` and `word2vec_chatbot_model_v2.pkl` in repo root.
- Streamlit port in use: `streamlit run app.py --server.port 8502`
- NLTK data issues: run the NLTK download snippet above.

## Next steps & tips
- Retrain with more examples for rarer concepts.
- Improve preprocessing (keep some punctuation, preserve code tokens) for better code-related answers.
- Consider a hybrid pipeline (fast retrieval + a small generative model) only if allowed by your requirements.

---

Built for teaching and demonstration — feel free to adapt and extend.
