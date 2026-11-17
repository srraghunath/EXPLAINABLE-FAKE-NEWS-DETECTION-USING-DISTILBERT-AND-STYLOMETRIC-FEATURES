# Explainable Fake News Detection

Flask application for fake-news classification using DistilBERT and LIME
explanations, with optional Groq and OpenRouter analysis.

## Setup

1. Create a virtual environment and install the dependencies:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. Provide trained model artifacts in `baseline_model/` and `robust_model/`.
   These files are intentionally excluded from Git because the model weights
   exceed GitHub's per-file size limit.

3. Store API credentials locally in `groq_api_key.txt` and
   `openrouter_api_key.txt`, or configure the providers through the
   application's supported environment variables. Credential files are
   intentionally ignored by Git.

4. Start the application:

   ```powershell
   python app.py
   ```
