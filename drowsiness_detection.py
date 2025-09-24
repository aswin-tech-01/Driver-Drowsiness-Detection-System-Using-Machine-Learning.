import cv2
import dlib
from scipy.spatial import distance
import pygame

# Initialize sound alert
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("alert.wav")  # keep an alert.wav file in repo

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye_points):
    A = distance.euclidean(eye_points[1], eye_points[5])
    B = distance.euclidean(eye_points[2], eye_points[4])
    C = distance.euclidean(eye_points[0], eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Load face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # download pre-trained model

# Indexes for left and right eye landmarks
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

# EAR threshold and consecutive frames
EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 20
counter = 0

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in LEFT_EYE]
        right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in RIGHT_EYE]

        # Compute EAR for both eyes
        left_EAR = eye_aspect_ratio(left_eye)
        right_EAR = eye_aspect_ratio(right_eye)
        avg_EAR = (left_EAR + right_EAR) / 2.0

        # Draw eyes
        for (x, y) in left_eye + right_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Check if EAR below threshold
        if avg_EAR < EAR_THRESHOLD:
            counter += 1
            if counter >= CONSEC_FRAMES:
                cv2.putText(frame, "DROWSINESS ALERT!", (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                alert_sound.play()
        else:
            counter = 0

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
