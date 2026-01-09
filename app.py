from flask import Flask, request, jsonify, render_template
import os
import traceback

app = Flask(__name__)

# Try to load model and tokenizer if they exist
MODEL_PATH = 'model.h5'
TOKENIZER_PATH = 'tokenizer.pickle'

model = None
tokenizer = None
maxlen = 100

try:
    if os.path.exists(MODEL_PATH):
        from tensorflow.keras.models import load_model
        model = load_model(MODEL_PATH)
        print('Loaded model from', MODEL_PATH)
    else:
        print('No model file found at', MODEL_PATH)
except Exception:
    print('Failed to load model:')
    traceback.print_exc()

try:
    if os.path.exists(TOKENIZER_PATH):
        import pickle
        with open(TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)
        print('Loaded tokenizer from', TOKENIZER_PATH)
    else:
        print('No tokenizer file found at', TOKENIZER_PATH)
except Exception:
    print('Failed to load tokenizer:')
    traceback.print_exc()


def preprocess(text):
    """Preprocess input text using tokenizer if available.

    If no tokenizer is found, returns None.
    """
    if tokenizer is None:
        return None
    try:
        sequences = tokenizer.texts_to_sequences([text])
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        padded = pad_sequences(sequences, maxlen=maxlen)
        return padded
    except Exception:
        traceback.print_exc()
        return None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    if not data or 'text' not in data:
        return jsonify({'error': 'Expected JSON with field "text"'}), 400

    text = data['text']
    x = preprocess(text)
    if model is None:
        return jsonify({'error': 'Model not loaded. Place `model.h5` in the project root.'}), 500
    if x is None:
        return jsonify({'error': 'Tokenizer not available or preprocessing failed. Place `tokenizer.pickle` or adapt preprocessing.'}), 500

    try:
        pred = model.predict(x)
        # Convert numpy arrays to Python lists for JSON serialization
        return jsonify({'prediction': pred.tolist()})
    except Exception:
        traceback.print_exc()
        return jsonify({'error': 'Model prediction failed.'}), 500


if __name__ == '__main__':
    # Run app directly for simplicity (development only)
    app.run(host='0.0.0.0', port=5000, debug=True)
