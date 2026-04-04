# app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os

from utils.predictor import GesturePredictor
from utils.sentence_logic import SentenceBuilder

app = Flask(__name__, static_folder='static')
CORS(app)

MODEL_PATH = "gesture_model.h5"

if os.path.exists(MODEL_PATH):
    predictor = GesturePredictor(MODEL_PATH)
    print("✅ Real model loaded")
else:
    predictor = None
    print("⚠️  No model found — running in dummy mode")

sentence_builder = SentenceBuilder(hold_frames=8, confidence_threshold=0.70)
conversation_history = []
camera_active = True


@app.route('/app')
def serve_frontend():
    return send_from_directory('static', 'index.html')

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "running", "model_loaded": predictor is not None})

@app.route('/predict', methods=['POST'])
def predict():
    if not camera_active:
        return jsonify({"letter": "", "confidence": 0, "sentence": sentence_builder.get_sentence(), "camera": "off"})
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
    return jsonify({"letter": letter, "confidence": round(confidence, 7), "sentence": sentence})

@app.route('/sentence/backspace', methods=['POST'])
def backspace():
    return jsonify({"sentence": sentence_builder.backspace()})

@app.route('/sentence/clear', methods=['POST'])
def clear():
    return jsonify({"sentence": sentence_builder.clear()})

@app.route('/sentence/get', methods=['GET'])
def get_sentence():
    return jsonify({"sentence": sentence_builder.get_sentence()})

@app.route('/sentence/space', methods=['POST'])
def add_space():
    sentence_builder.sentence += " "
    return jsonify({"sentence": sentence_builder.get_sentence()})

@app.route('/speak', methods=['POST'])
def speak():
    data = request.get_json()
    text = data.get('text', '') or sentence_builder.get_sentence()
    if not text:
        return jsonify({"error": "No text to speak"}), 400
    conversation_history.append(text)
    return jsonify({"status": "spoken", "text": text})

@app.route('/history', methods=['GET'])
def history():
    return jsonify({"history": conversation_history})

@app.route('/history/clear', methods=['POST'])
def clear_history():
    conversation_history.clear()
    return jsonify({"status": "cleared"})

@app.route('/camera/on', methods=['POST'])
def camera_on():
    global camera_active
    camera_active = True
    return jsonify({"camera": "on"})

@app.route('/camera/off', methods=['POST'])
def camera_off():
    global camera_active
    camera_active = False
    return jsonify({"camera": "off"})

@app.route('/camera/status', methods=['GET'])
def camera_status():
    return jsonify({"camera": "on" if camera_active else "off"})

@app.route('/status', methods=['GET'])
def status():
    return jsonify({"server": "running", "model_loaded": predictor is not None,
                    "current_sentence": sentence_builder.get_sentence(),
                    "total_spoken": len(conversation_history),
                    "camera": "on" if camera_active else "off"})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)