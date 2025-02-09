import cv2
import mediapipe as mp
import numpy as np
import os

# Параметры пользователя
USER_ID = 0
NAME = "Rinat"
SURNAME = "Khamidullov"

# Инициализация распознавания лиц
path = os.path.dirname(os.path.abspath(__file__))
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(os.path.join(path, "trainer", "trainer.yml"))

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Инициализация распознавания руки (mediapipe)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Открытие камеры
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

def count_fingers(hand_landmarks):
    """Функция считает количество поднятых пальцев."""
    fingers = [0, 0, 0, 0, 0]  # [большой, указательный, средний, безымянный, мизинец]
    tips = [4, 8, 12, 16, 20]  # ID концов пальцев
    for i in range(1, 5):
        if hand_landmarks.landmark[tips[i]].y < hand_landmarks.landmark[tips[i] - 2].y:
            fingers[i] = 1  # Палец поднят
    return sum(fingers)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))

    # Обнаружение лица
    label = "Unknown"
    for (x, y, w, h) in faces:
        face_id, conf = recognizer.predict(gray[y:y+h, x:x+w])
        if conf < 50 and face_id == USER_ID:
            label = "Face detected"

        cv2.rectangle(frame, (x, y), (x + w, y + h), (225, 0, 0), 2)

    # Обнаружение рук
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            fingers_count = count_fingers(hand_landmarks)

            # Проверяем, какую руку видит камера
            hand_type = "Левая" if hand_landmarks.landmark[0].x < 0.5 else "Правая"

            # Подписываем имя/фамилию в зависимости от поднятых пальцев
            if fingers_count == 1:
                if label == "Face detected":
                    label = NAME
            elif fingers_count == 2:
                if label == "Face detected":
                    label = SURNAME

    # Вывод текста
    for (x, y, w, h) in faces:
        cv2.putText(frame, label, (x, y - 10), font, 1, (0, 255, 0), 2)

    cv2.imshow("Face & Hand Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
