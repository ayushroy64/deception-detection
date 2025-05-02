import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

# Load MobileNetV2 (exclude top classifier layer)
mobilenet = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

# Parameters
SEQUENCE_LENGTH = 30  # number of frames per sequence
FRAME_SIZE = (224, 224)

def extract_mobilenet_sequence(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.resize(frame, FRAME_SIZE)
        frame = frame / 255.0
        frames.append(frame)
    
    cap.release()
    
    if len(frames) < SEQUENCE_LENGTH:
        return None  # skip short videos
    
    frames = np.array(frames)
    sequences = []
    
    for i in range(0, len(frames) - SEQUENCE_LENGTH + 1, SEQUENCE_LENGTH):
        clip = frames[i:i + SEQUENCE_LENGTH]
        # Extract embeddings for each frame in the sequence
        embeddings = mobilenet.predict(clip, verbose=0)
        sequences.append(embeddings)
    
    return sequences

# Load labels
df = pd.read_csv("data.csv")
base_path_deception = "C:/Reet/College/FINAL_USC/csci467/project/Deceptive"
base_path_truth = "C:/Reet/College/FINAL_USC/csci467/project/Truthful"

X, y = [], []

for _, row in df.iterrows():
    filename = row["id"]
    label = 1 if row["class"] == "deceptive" else 0
    video_path = os.path.join(base_path_deception if label == 1 else base_path_truth, filename)
    
    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        continue

    sequence_embeddings = extract_mobilenet_sequence(video_path)
    
    if sequence_embeddings:
        for seq in sequence_embeddings:
            X.append(seq)
            y.append(label)

X = np.array(X)
y = np.array(y)

# Split the dataset
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# LSTM Model
model = Sequential([
    LSTM(128, return_sequences=False, input_shape=(SEQUENCE_LENGTH, 1280)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=15, batch_size=8, validation_data=(X_dev, y_dev))

# Evaluate
y_train_pred = (model.predict(X_train) > 0.5).astype("int32")
y_dev_pred = (model.predict(X_dev) > 0.5).astype("int32")
y_test_pred = (model.predict(X_test) > 0.5).astype("int32")

train_acc = accuracy_score(y_train, y_train_pred)
dev_acc = accuracy_score(y_dev, y_dev_pred)
test_acc = accuracy_score(y_test, y_test_pred)

train_f1 = f1_score(y_train, y_train_pred)
dev_f1 = f1_score(y_dev, y_dev_pred)
test_f1 = f1_score(y_test, y_test_pred)

print(f"Training Accuracy: {train_acc:.2f}, F1 Score: {train_f1:.2f}")
print(f"Development Accuracy: {dev_acc:.2f}, F1 Score: {dev_f1:.2f}")
print(f"Testing Accuracy: {test_acc:.2f}, F1 Score: {test_f1:.2f}")
