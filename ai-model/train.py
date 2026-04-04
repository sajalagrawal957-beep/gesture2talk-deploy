import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. Load the data you just created
print("📂 Loading landmarks.csv...")
data = pd.read_csv("landmarks.csv", header=None)

# X = coordinates (42 columns now!), y = the letter (last column)
X = data.iloc[:, :-1].values.astype(np.float32)
y = data.iloc[:, -1].values.astype(str)

# Normalize (Updated for 2D: 42 columns, 2 coordinates per point)
X = X.reshape(-1, 21, 2)  # Changed 3 to 2
X = X - X[:, 0:1, :]
X = X / (np.max(np.abs(X), axis=(1,2), keepdims=True) + 1e-6)
X = X.reshape(-1, 42)     # Changed 63 to 42

# 2. Convert letters (A, B, C) into numbers (0, 1, 2)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(np.unique(y_encoded))

# 3. Split: 80% to learn, 20% to test if it learned correctly
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 4. Create the Neural Network
model = Sequential([
    # ✅ Tell the AI to expect 42 inputs instead of 63
    tf.keras.Input(shape=(42,)),

    Dense(256, activation='relu'),
    Dropout(0.3),

    Dense(128, activation='relu'),
    Dropout(0.3),

    Dense(64, activation='relu'),

    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 5. Train the AI
print("⚙️ Training the AI... watch the 'accuracy' go up!")
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# 6. Save the results
model.save("gesture_model.h5")
np.save('classes.npy', label_encoder.classes_)
print("✅ Success! 'gesture_model.h5' is ready. This is your AI Brain.")