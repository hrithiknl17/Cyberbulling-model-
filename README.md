# Minimal Flask API for Cyberbullying LSTM

This small app provides a minimal web UI and a `/predict` JSON endpoint to run a Keras model saved as `model.h5`.

Quick start (PowerShell):

1. Create and activate a virtual environment (optional but recommended):

```powershell
python -m venv .venv
; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Place your model and tokenizer:
- Put your Keras model at `model.h5` in the project root.
- If your preprocessing uses a Keras `Tokenizer`, pickle it to `tokenizer.pickle` in the project root.

4. Run the app:

```powershell
python app.py
```

5. Open http://localhost:5000 in your browser and test the UI, or call the API:

Example request using `curl` (PowerShell):

```powershell
curl -Method Post -Uri http://localhost:5000/predict -ContentType 'application/json' -Body '{"text":"you are awful"}'
```

Notes:
- The `app.py` will try to load `model.h5` and `tokenizer.pickle`. If you don't have a tokenizer file, either generate one from your training pipeline or adapt `preprocess()` in `app.py` to match the notebook preprocessing.
- This scaffold is for development and testing only. For production, containerize and use a production WSGI server.
