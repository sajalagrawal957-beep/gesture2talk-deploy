# utils/predictor.py

import numpy as np

LABELS = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']

class GesturePredictor:
    def __init__(self, model_path):
        print("Loading model...")
        import tensorflow as tf
        # Fix for version mismatch
        try:
            self.model = tf.keras.models.load_model(model_path)
        except Exception:
            self.model = tf.keras.models.load_model(
                model_path, 
                compile=False,
                custom_objects=None,
                options=tf.saved_model.LoadOptions(
                    experimental_io_device='/job:localhost'
                )
            )
        print("Model loaded!")

    def predict(self, landmarks):
        input_data = np.array(landmarks).reshape(1, -1)
        predictions = self.model.predict(input_data, verbose=0)
        best_index = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][best_index])
        letter = LABELS[best_index]
        return letter, confidence