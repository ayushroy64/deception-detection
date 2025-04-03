import os
import cv2
import csv
import mediapipe as mp
import numpy as np
from pathlib import Path
from collections import deque
from sklearn.metrics import classification_report

# Define paths (Linux format)
BASE_PATH = Path("./data/Clips")  
ANNOTATIONS_PATH = Path("./data/annotations.csv")  

# Load annotations
if not ANNOTATIONS_PATH.exists():
    raise FileNotFoundError("Annotations file not found!")

annotations = {}
deceptive_videos = []
truthful_videos = []

with open(ANNOTATIONS_PATH, "r") as f:
    reader = csv.DictReader(f)  
    for row in reader:
        video_file = row["id"].strip()
        label = row["class"].strip().lower()
        
        # Store based on label
        if label == "deceptive" and len(deceptive_videos) < 10:
            deceptive_videos.append(video_file)
        elif label == "truthful" and len(truthful_videos) < 10:
            truthful_videos.append(video_file)
        
        annotations[video_file] = label
        if len(deceptive_videos) >= 10 and len(truthful_videos) >= 10:
            break

test_videos = deceptive_videos + truthful_videos
print(f"Evaluating on {len(test_videos)} test videos ({len(deceptive_videos)} deceptive, {len(truthful_videos)} truthful).")

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Feature Extraction Functions
def calculate_ear(landmarks, indices):
    if len(indices) < 6:
        return 0.3  # Default to a normal EAR value if invalid input
    try:
        A = np.linalg.norm(np.array(landmarks[1]) - np.array(landmarks[5]))
        B = np.linalg.norm(np.array(landmarks[2]) - np.array(landmarks[4]))
        C = np.linalg.norm(np.array(landmarks[0]) - np.array(landmarks[3]))
        return (A + B) / (2.0 * C) if C != 0 else 0.3  # Avoid division by zero
    except:
        return 0.3

def calculate_mar(landmarks, indices):
    if len(indices) < 8:
        return 0.3  
    try:
        A = np.linalg.norm(np.array(landmarks[1]) - np.array(landmarks[7]))
        B = np.linalg.norm(np.array(landmarks[2]) - np.array(landmarks[6]))
        C = np.linalg.norm(np.array(landmarks[3]) - np.array(landmarks[5]))
        D = np.linalg.norm(np.array(landmarks[0]) - np.array(landmarks[4]))
        return (A + B + C) / (2.0 * D) if D != 0 else 0.3
    except:
        return 0.3

def detect_blinks(ear, prev_ear, threshold=0.2):
    return prev_ear > threshold and ear < threshold  

def calculate_asymmetry(landmarks):
    try:
        left = np.linalg.norm(np.array(landmarks[61]) - np.array(landmarks[291]))
        right = np.linalg.norm(np.array(landmarks[291]) - np.array(landmarks[61]))
        return abs(left - right)
    except:
        return 0  

# Indices
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
MOUTH_INDICES = [61, 146, 91, 181, 84, 17, 314, 405]

def process_video(video_path):
    cap = cv2.VideoCapture(str(video_path))

    # Rolling History
    window_size = 10  
    ear_history = deque(maxlen=window_size)
    mar_history = deque(maxlen=window_size)
    asymmetry_history = deque(maxlen=window_size)
    blink_history = deque(maxlen=window_size)
    head_movement_history = deque(maxlen=window_size)
    deception_score_history = deque(maxlen=5)  

    prev_ear = 0.3  
    blinks = 0  
    frame_count = 0  
    prev_head_x, prev_head_y = None, None  

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = [(l.x * frame.shape[1], l.y * frame.shape[0]) for l in face_landmarks.landmark]

                # Compute Features
                ear = (calculate_ear(landmarks, LEFT_EYE_INDICES) + calculate_ear(landmarks, RIGHT_EYE_INDICES)) / 2.0
                mar = calculate_mar(landmarks, MOUTH_INDICES)
                asymmetry = calculate_asymmetry(landmarks)

                # Blink Detection
                if detect_blinks(ear, prev_ear):
                    blinks += 1  
                prev_ear = ear  

                # Head Movement
                head_x, head_y = landmarks[0]  
                if prev_head_x is not None and prev_head_y is not None:
                    head_movement = np.linalg.norm([head_x - prev_head_x, head_y - prev_head_y])
                else:
                    head_movement = 0  
                prev_head_x, prev_head_y = head_x, head_y  

                # Store in Rolling History
                ear_history.append(ear)
                mar_history.append(mar)
                asymmetry_history.append(asymmetry)
                blink_history.append(blinks)
                head_movement_history.append(head_movement)

                # Compute Adaptive Thresholds (with safeguards)
                avg_ear = np.mean(ear_history) if ear_history else 0.3
                avg_mar = np.mean(mar_history) if mar_history else 0.3
                avg_asymmetry = np.mean(asymmetry_history) if asymmetry_history else 0.0
                avg_blinks = np.mean(blink_history) if blink_history else 5
                avg_head_movement = np.mean(head_movement_history) if head_movement_history else 0.0

                # Apply Weighted Scoring System
                deception_score = 0
                if ear < avg_ear * 0.85:
                    deception_score += 2  
                if mar > avg_mar * 1.08:
                    deception_score += 2  
                if asymmetry > avg_asymmetry * 1.1:
                    deception_score += 1  
                if avg_blinks < 3 or avg_blinks > 20:
                    deception_score += 2  
                if avg_head_movement > 15:
                    deception_score += 2  

                deception_score_history.append(deception_score)

                # Visualization
                cv2.putText(frame, f"Deception Score: {deception_score}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("Processing Video", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    cv2.destroyAllWindows()

    # Final Prediction
    total_score = sum(deception_score_history)
    return "deceptive" if total_score >= 3 else "truthful"

# Evaluate model
y_true, y_pred = [], []
for video_name in test_videos:
    video_path = BASE_PATH / ("Deceptive" if video_name in deceptive_videos else "Truthful") / video_name
    if not video_path.exists():
        continue
    prediction = process_video(video_path)
    y_true.append(annotations[video_name])
    y_pred.append(prediction)

print("\n=== Model Performance Metrics ===")
print(classification_report(y_true, y_pred, target_names=["truthful", "deceptive"]))
