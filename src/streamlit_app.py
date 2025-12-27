import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os
import ast
from PIL import Image

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.keras')
INDICES_PATH = os.path.join(BASE_DIR, 'models', 'class_indices.txt')

# Load Model (Cached to avoid reloading on every interaction)
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_labels():
    try:
        with open(INDICES_PATH, 'r') as f:
            indices = ast.literal_eval(f.read())
            return {v: k for k, v in indices.items()}
    except FileNotFoundError:
        return {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

st.title("😊 Facial Mood Detection")
st.write("This is a web-ready version of the mood detection capstone project.")

model = load_model()
labels_map = load_labels()

if model is None:
    st.error("Model not found! Please run `python src/train.py` locally first to generate the model, or upload a pre-trained `best_model.keras`.")
else:
    # Sidebar options
    st.sidebar.title("Options")
    input_mode = st.sidebar.radio("Choose Input Mode", ["Webcam Snapshot", "Upload Image"])

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    img_input = None

    if input_mode == "Webcam Snapshot":
        img_file_buffer = st.camera_input("Take a picture")
        if img_file_buffer is not None:
            # To read image file buffer with OpenCV:
            bytes_data = img_file_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            img_input = cv2_img

    elif input_mode == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            cv2_img = cv2.imdecode(file_bytes, 1)
            img_input = cv2_img
            st.image(cv2_img, channels="BGR", caption="Uploaded Image")

    if img_input is not None:
        if st.button("Analyze Mood"):
            gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) == 0:
                st.warning("No face detected in the image.")
            else:
                for (x, y, w, h) in faces:
                    # Draw rect on original image for display
                    cv2.rectangle(img_input, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    
                    # Preprocess for model
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_gray = cv2.resize(roi_gray, (48, 48))
                    roi_gray = roi_gray.astype('float') / 255.0
                    roi_gray = np.expand_dims(roi_gray, axis=0)
                    roi_gray = np.expand_dims(roi_gray, axis=-1)

                    # Predict
                    prediction = model.predict(roi_gray)
                    max_index = int(np.argmax(prediction))
                    predicted_label = labels_map.get(max_index, "Unknown")
                    confidence = prediction[0][max_index]

                    st.success(f"Detected Mood: **{predicted_label}** ({confidence*100:.1f}%)")
                
                st.image(img_input, channels="BGR", caption="Processed Image")
