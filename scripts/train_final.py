import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

print("Loading data...")
df = pd.read_csv(r"C:\Users\aaa\SignBridgeProject\data\landmarks.csv")
X = df.drop("label", axis=1).values
y = df["label"].values

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
np.save(r"C:\Users\aaa\SignBridgeProject\models\classes.npy", encoder.classes_)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

print("Building model...")
model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(len(encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Training...")
model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=1)

print("Saving weights...")
weights = model.get_weights()
np.savez(r"C:\Users\aaa\SignBridgeProject\models\model_weights.npz", 
         w1=weights[0], b1=weights[1], w2=weights[2], b2=weights[3], w3=weights[4], b3=weights[5])

print("✅ Model trained and saved perfectly!")