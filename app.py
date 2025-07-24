import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from datetime import datetime

# Load the pre-trained model
model = load_model('mood_model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def suggest_activity(emotion):
    suggestions = {
        'Happy': 'Keep smiling! Maybe share your joy with someone.',
        'Sad': 'Listen to your favorite music or take a walk.',
        'Angry': 'Take deep breaths, try meditation.',
        'Neutral': 'Keep going! Maybe do something fun.',
        'Fear': 'Talk to a friend or write your thoughts down.',
        'Surprise': 'Write about what surprised you!',
        'Disgust': 'Take a break, drink water.',
    }
    return suggestions.get(emotion, "Just be yourself!")

def log_mood(emotion):
    with open("mood_log.txt", "a") as f:
        f.write(f"{datetime.now()} - Detected mood: {emotion}\n")

# Start video capture
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = model.predict(roi, verbose=0)[0]
            emotion = emotion_labels[preds.argmax()]
            label_position = (x, y)
            cv2.putText(frame, emotion, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            log_mood(emotion)
            print(f"Suggested Activity: {suggest_activity(emotion)}")

    cv2.imshow("Real-Time Mood Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()