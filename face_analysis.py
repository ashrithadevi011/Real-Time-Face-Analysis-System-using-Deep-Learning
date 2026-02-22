import cv2
from deepface import DeepFace

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Analyze face for age, gender, emotion
        result = DeepFace.analyze(
            frame,
            actions=['age', 'gender', 'emotion'],
            enforce_detection=False
        )

        age = result[0]['age']
        gender = result[0]['dominant_gender']
        emotion = result[0]['dominant_emotion']

        text = f"{gender}, {emotion}, Age: {age}"

        cv2.putText(frame, text, (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

    except:
        pass

    cv2.imshow("Face Analysis - Age, Gender, Emotion", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()