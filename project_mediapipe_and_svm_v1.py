import os
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Initialize Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Function to extract facial landmarks from video
def extract_landmarks(video_src):
    capt = cv2.VideoCapture(video_src)
    frame_features = []
    
    while capt.isOpened():
        ret, frame = capt.read()
        if not ret:
            break  
        
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frameRGB)
        
        if results.multi_face_landmarks:
            faceLandmarks = results.multi_face_landmarks[0]
            landmarks = []
            for landmark in faceLandmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            frame_features.append(landmarks)
    
    capt.release()
    
    if len(frame_features) > 0:
        return np.mean(frame_features, axis=0) 
    else:
        return None 


# Load dataset
base_path_deception = "C:\Reet\College\FINAL_USC\csci467\project\Deceptive"  
base_path_truth = "C:\Reet\College\FINAL_USC\csci467\project\Truthful"  

# Load dataset
df = pd.read_csv("data.csv")  

# Extract Features and Labels
X, y = [], []

for _, row in df.iterrows():
    filename = row["id"]
    label = 1 if row["class"] == "deceptive" else 0 
    
    # Dealing with file path issues 
    video_path = os.path.join(base_path_deception, filename) if label == 1 else os.path.join(base_path_truth, filename)

    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        continue

    features = extract_landmarks(video_path)
    if features is not None:
        X.append(features)
        y.append(label)

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

# Standardize Features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Stratified Split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

# Train SVM Model
svm_model = SVC(kernel="rbf", C=1.0, gamma="scale")
svm_model.fit(X_train, y_train)

# Predictions for Each Set
y_train_pred = svm_model.predict(X_train)
y_dev_pred = svm_model.predict(X_dev)
y_test_pred = svm_model.predict(X_test)

# Compute Accuracy
train_acc = accuracy_score(y_train, y_train_pred)
dev_acc = accuracy_score(y_dev, y_dev_pred)
test_acc = accuracy_score(y_test, y_test_pred)

# Compute F1 Score
train_f1 = f1_score(y_train, y_train_pred)
dev_f1 = f1_score(y_dev, y_dev_pred)
test_f1 = f1_score(y_test, y_test_pred)

# Print Evaluation Metrics
print(f"Training Accuracy: {train_acc:.2f}, F1 Score: {train_f1:.2f}")
print(f"Development Accuracy: {dev_acc:.2f}, F1 Score: {dev_f1:.2f}")
print(f"Testing Accuracy: {test_acc:.2f}, F1 Score: {test_f1:.2f}")

