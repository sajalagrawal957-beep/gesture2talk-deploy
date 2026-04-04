# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pyttsx3

from utils.predictor import GesturePredictor
from utils.sentence_logic import SentenceBuilder

app = Flask(__name__)
CORS(app)

# ── Load model ONCE when server starts ──────────────────────────────────────
MODEL_PATH = "gesture_model.h5"

if os.path.exists(MODEL_PATH):
    predictor = GesturePredictor(MODEL_PATH)
    print("✅ Real model loaded")
else:
    predictor = None
    print("⚠️  No model found — running in dummy mode")

sentence_builder = SentenceBuilder(hold_frames=15, confidence_threshold=0.75)


# ── Route 1: Health check ────────────────────────────────────────────────────
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "running",
        "model_loaded": predictor is not None
    })


# ── Route 2: Predict ─────────────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data or 'landmarks' not in data:
        return jsonify({"error": "No landmarks provided"}), 400

    landmarks = data['landmarks']

    if len(landmarks) != 42:
        return jsonify({"error": f"Expected 42 landmarks, got {len(landmarks)}"}), 400

    if predictor:
        letter, confidence = predictor.predict(landmarks)
    else:
        letter = "A" 
        confidence = 0.99

    sentence = sentence_builder.update(letter, confidence)

    return jsonify({
        "letter": letter,
        "confidence": round(confidence, 7),
        "sentence": sentence
    })


# ── Route 3: Backspace ───────────────────────────────────────────────────────
@app.route('/sentence/backspace', methods=['POST'])
def backspace():
    sentence = sentence_builder.backspace()
    return jsonify({"sentence": sentence})


# ── Route 4: Clear ───────────────────────────────────────────────────────────
@app.route('/sentence/clear', methods=['POST'])
def clear():
    sentence = sentence_builder.clear()
    return jsonify({"sentence": sentence})


# ── Route 5: Get sentence ────────────────────────────────────────────────────
@app.route('/sentence/get', methods=['GET'])
def get_sentence():
    return jsonify({"sentence": sentence_builder.get_sentence()})

# ── Route 6: Speak ───────────────────────────────────────────────────────────
# ── Route 6: Speak ───────────────────────────────────────────────────────────
import threading

@app.route('/speak', methods=['POST'])
def speak():
    data = request.get_json()

    text = data.get('text', '')

    if not text:
        text = sentence_builder.get_sentence()

    if not text:
        return jsonify({"error": "No text to speak"}), 400
    
    conversation_history.append(text)

    def speak_text(t):
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 1.0)
            engine.say(t)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print(f"TTS Error: {e}")

    # Run in background thread so Flask doesn't crash
    thread = threading.Thread(target=speak_text, args=(text,))
    thread.start()

    return jsonify({
        "status": "spoken",
        "text": text
    })

# ── BONUS: Conversation History ──────────────────────────────────────────────
conversation_history = []

@app.route('/history', methods=['GET'])
def history():
    return jsonify({"history": conversation_history})

@app.route('/history/clear', methods=['POST'])
def clear_history():
    conversation_history.clear()
    return jsonify({"status": "cleared"})

# ── BONUS: Status Page ────────────────────────────────────────────────────────
@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        "server": "running",
        "model_loaded": predictor is not None,
        "current_sentence": sentence_builder.get_sentence(),
        "total_spoken": len(conversation_history)
    })


# ── Run ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, port=5000)
