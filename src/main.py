import cv2
import numpy as np
import tensorflow as tf
import os
import ast

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.keras')
INDICES_PATH = os.path.join(BASE_DIR, 'models', 'class_indices.txt')

def load_labels():
    try:
        with open(INDICES_PATH, 'r') as f:
            indices = ast.literal_eval(f.read())
            # Invert the dictionary to map index -> label
            return {v: k for k, v in indices.items()}
    except FileNotFoundError:
        print("Class indices file not found. Using default list.")
        return {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

def main():
    if not os.path.exists(MODEL_PATH):
        print("Model file not found! Please run src/train.py first.")
        return

    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    labels_map = load_labels()
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam")
        return

    print("Starting inference... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # ROI
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = roi_gray.astype('float') / 255.0
            roi_gray = np.expand_dims(roi_gray, axis=0) # batch dim
            roi_gray = np.expand_dims(roi_gray, axis=-1) # channel dim

            # Predict
            prediction = model.predict(roi_gray, verbose=0)
            max_index = int(np.argmax(prediction))
            predicted_label = labels_map.get(max_index, "Unknown")
            confidence = prediction[0][max_index]

            # Draw
            color = (0, 255, 0) if predicted_label == 'Happy' else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{predicted_label} ({confidence:.2f})", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow('Facial Mood Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
