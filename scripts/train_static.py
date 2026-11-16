import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import os

# ─── CONFIG ────────────────────────────────────────────────
CSV_PATH = r"C:\Users\aaa\SignBridgeProject\data\landmarks.csv"
MODEL_SAVE_PATH = r"C:\Users\aaa\SignBridgeProject\models\static_model.h5"
# ───────────────────────────────────────────────────────────

print("1. Loading data...")
df = pd.read_csv(CSV_PATH)

# Separate features (X) and labels (y)
X = df.drop("label", axis=1).values
y = df["label"].values

# Convert letters (A, B, C...) into numbers (0, 1, 2...)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Save the encoder classes so we can translate numbers back to letters later
np.save(r"C:\Users\aaa\SignBridgeProject\models\classes.npy", encoder.classes_)

print("2. Splitting data into Training (80%) and Testing (20%)...")
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

print("3. Building the Neural Network...")
model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)), # 63 input features (21 landmarks x 3)
    Dropout(0.3),                                     # Prevents memorizing
    Dense(64, activation='relu'),
    Dense(len(encoder.classes_), activation='softmax') # Output layer (24 letters)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("4. Training the model (This will take 1-2 minutes)...")
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test), batch_size=32)

print("5. Evaluating model...")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n🎉 Final Test Accuracy: {accuracy * 100:.2f}%")

print("6. Saving the model...")
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
model.save(MODEL_SAVE_PATH)
print(f"📁 Model saved to {MODEL_SAVE_PATH}")