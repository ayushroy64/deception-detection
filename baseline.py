import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# EAR Calculation Function
def calculate_ear(landmarks, eye_indices):
    """ Compute Eye Aspect Ratio (EAR) to detect eye openness changes. """
    A = np.linalg.norm(np.array(landmarks[1]) - np.array(landmarks[5]))  # Vertical
    B = np.linalg.norm(np.array(landmarks[2]) - np.array(landmarks[4]))  # Vertical
    C = np.linalg.norm(np.array(landmarks[0]) - np.array(landmarks[3]))  # Horizontal
    return (A + B) / (2.0 * C)  # EAR formula

# MAR Calculation Function
def calculate_mar(landmarks, mouth_indices):
    """ Compute Mouth Aspect Ratio (MAR) to track mouth movement intensity. """
    A = np.linalg.norm(np.array(landmarks[1]) - np.array(landmarks[7]))  
    B = np.linalg.norm(np.array(landmarks[2]) - np.array(landmarks[6]))  
    C = np.linalg.norm(np.array(landmarks[3]) - np.array(landmarks[5]))  
    D = np.linalg.norm(np.array(landmarks[0]) - np.array(landmarks[4]))  
    return (A + B + C) / (2.0 * D)  # MAR formula

# Deception Feature Extraction
def detect_flared_nostrils(landmarks):
    """ Detect nostril flaring based on relative nose width changes. """
    nose_width = np.linalg.norm(np.array(landmarks[49]) - np.array(landmarks[279]))  
    return nose_width  

def detect_lip_biting(landmarks):
    """ Detect lip compression (possible self-restraint cue). """
    lip_distance = np.linalg.norm(np.array(landmarks[13]) - np.array(landmarks[14]))  
    return lip_distance  

def detect_forced_smile(landmarks):
    """ Detects asymmetry in smile by comparing mouth corner movements. """
    left_smile = np.linalg.norm(np.array(landmarks[61]) - np.array(landmarks[291]))  
    right_smile = np.linalg.norm(np.array(landmarks[291]) - np.array(landmarks[61]))  
    return abs(left_smile - right_smile)  

# Landmark Indices
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
MOUTH_INDICES = [61, 146, 91, 181, 84, 17, 314, 405]

# Rolling Window for Temporal Analysis
window_size = 10  
ear_history = deque(maxlen=window_size)  
mar_history = deque(maxlen=window_size)  
nostril_history = deque(maxlen=window_size)  
lip_bite_history = deque(maxlen=window_size)  
smile_history = deque(maxlen=window_size)  
deception_score_history = deque(maxlen=5)  # Keep track of last 5 frames' scores  

cap = cv2.VideoCapture(0)  # Webcam feed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract facial landmark points
            landmarks = [(l.x * frame.shape[1], l.y * frame.shape[0]) for l in face_landmarks.landmark]

            # Compute Features
            ear = (calculate_ear(landmarks, LEFT_EYE_INDICES) + calculate_ear(landmarks, RIGHT_EYE_INDICES)) / 2.0
            mar = calculate_mar(landmarks, MOUTH_INDICES)
            nostrils_flared = detect_flared_nostrils(landmarks)
            lip_biting = detect_lip_biting(landmarks)
            forced_smile = detect_forced_smile(landmarks)

            # Store in Rolling History
            ear_history.append(ear)
            mar_history.append(mar)
            nostril_history.append(nostrils_flared)
            lip_bite_history.append(lip_biting)
            smile_history.append(forced_smile)

            # Compute Adaptive Thresholds
            avg_ear = np.mean(ear_history)
            avg_mar = np.mean(mar_history)
            avg_nostrils = np.mean(nostril_history)
            avg_lip_bite = np.mean(lip_bite_history)
            avg_smile = np.mean(smile_history)

            # Apply Weighted Scoring System Instead of Hard Thresholds
            deception_score = 0
            if ear < avg_ear * 0.9:  # Sudden decrease in EAR
                deception_score += 2
            if mar > avg_mar * 1.1:  # Sudden increase in MAR
                deception_score += 2
            if nostrils_flared > avg_nostrils * 1.05:  # Sudden nostril flare
                deception_score += 1
            if lip_biting < avg_lip_bite * 0.9:  # Lips pressed together
                deception_score += 1
            if avg_smile > 0.03:  # Large asymmetry in smile
                deception_score += 1

            # Track deception score history
            deception_score_history.append(deception_score)

            # Determine Status with Persistence Check
            if sum(deception_score_history) >= 8:  # If deception score persists over last 5 frames
                status = "DECEPTIVE"
                color = (0, 0, 255)
            elif sum(deception_score_history) >= 5:
                status = "UNCERTAIN"
                color = (0, 255, 255)
            else:
                status = "TRUTHFUL"
                color = (0, 255, 0)

            # Logging for Debugging
            print(f"EAR: {ear:.2f}, MAR: {mar:.2f}, Score: {deception_score}, Status: {status}")

            # Display Features & Status
            cv2.putText(frame, f'EAR: {ear:.2f}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f'MAR: {mar:.2f}', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f'Score: {deception_score}', (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, status, (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Show Video Feed
    cv2.imshow("Enhanced Deception Detection - Baseline", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()