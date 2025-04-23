import os
import cv2
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- Load MobileNetV2 ---
mobilenet = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
mobilenet.trainable = False  # Freeze model

# --- Extract features using MobileNetV2 from a video ---
def extract_mobilenet_features(video_path, num_frames=20):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        return None

    indices = np.linspace(0, frame_count - 1, num=num_frames).astype(int)
    features = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (224, 224))
        frame = preprocess_input(frame)
        frame = np.expand_dims(frame, axis=0)
        embedding = mobilenet.predict(frame, verbose=0)
        pooled = np.mean(embedding, axis=(1, 2))  # Global average pooling
        features.append(pooled[0])

    cap.release()

    if features:
        return np.mean(features, axis=0)
    else:
        return None

# --- File paths ---
base_path_deception = r"C:\Reet\College\FINAL_USC\csci467\project\Deceptive"
base_path_truth = r"C:\Reet\College\FINAL_USC\csci467\project\Truthful"
csv_path = "data.csv"  # Same format as before

# --- Load dataset from CSV ---
df = pd.read_csv(csv_path)

X, y = [], []

for _, row in df.iterrows():
    filename = row["id"]
    label = 1 if row["class"] == "deceptive" else 0

    # Full video path
    video_path = os.path.join(base_path_deception if label == 1 else base_path_truth, filename)

    if not os.path.exists(video_path):
        print(f"[Warning] File not found: {video_path}")
        continue

    features = extract_mobilenet_features(video_path)
    if features is not None:
        X.append(features)
        y.append(label)

# --- Convert and scale ---
X = np.array(X)
y = np.array(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# --- Split dataset ---
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

# --- Train SVM ---
svm_model = SVC(kernel="rbf", C=1.0, gamma="scale")
svm_model.fit(X_train, y_train)

# --- Evaluate ---
def evaluate(model, X, y, name="Set"):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    print(f"{name} Accuracy: {acc:.2f}, F1 Score: {f1:.2f}")
    return acc, f1

evaluate(svm_model, X_train, y_train, "Train")
evaluate(svm_model, X_dev, y_dev, "Dev")
evaluate(svm_model, X_test, y_test, "Test")
