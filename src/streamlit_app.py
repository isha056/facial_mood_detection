import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os
import sys
import ast
from PIL import Image

# Add src directory to path for custom layer import
sys.path.insert(0, os.path.dirname(__file__))

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.keras')
INDICES_PATH = os.path.join(BASE_DIR, 'models', 'class_indices.txt')

# Import custom layer at module level
try:
    from model_transfer import GrayscaleToRGB
    CUSTOM_OBJECTS = {'GrayscaleToRGB': GrayscaleToRGB}
except ImportError:
    CUSTOM_OBJECTS = None

# Emotion styling - emoji and colors
EMOTION_STYLES = {
    'angry': {'emoji': '😠', 'color': '#FF4B4B'},
    'disgust': {'emoji': '🤢', 'color': '#9B59B6'},
    'fear': {'emoji': '😨', 'color': '#E67E22'},
    'happy': {'emoji': '😊', 'color': '#2ECC71'},
    'neutral': {'emoji': '😐', 'color': '#95A5A6'},
    'sad': {'emoji': '😢', 'color': '#3498DB'},
    'surprise': {'emoji': '😲', 'color': '#F39C12'},
}

# Page config
st.set_page_config(
    page_title="Facial Mood Detection",
    page_icon="😊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        color: #666;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .emotion-result {
        text-align: center;
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
    }
    
    .emotion-emoji {
        font-size: 5rem;
        margin-bottom: 1rem;
    }
    
    .emotion-label {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load Model (Cached)
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH, custom_objects=CUSTOM_OBJECTS)

@st.cache_data
def load_labels():
    try:
        with open(INDICES_PATH, 'r') as f:
            indices = ast.literal_eval(f.read())
            return {v: k for k, v in indices.items()}
    except FileNotFoundError:
        return {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Header
st.markdown('<h1 class="main-header">🎭 Facial Mood Detection</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered emotion recognition using Deep Learning & Computer Vision</p>', unsafe_allow_html=True)

# Load resources
model = load_model()
labels_map = load_labels()

if model is None:
    st.error("⚠️ Model not found! Please train the model first using `python src/train_transfer.py`")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("### 🎛️ Settings")
    st.markdown("---")
    
    input_mode = st.radio(
        "📷 Input Mode",
        ["📸 Webcam Snapshot", "📁 Upload Image"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown(f"**Input Size:** {model.input_shape[1]}x{model.input_shape[2]}")
    st.markdown(f"**Classes:** {len(labels_map)}")
    
    st.markdown("---")
    st.markdown("### 🎭 Emotions")
    for emotion, style in EMOTION_STYLES.items():
        st.markdown(f"{style['emoji']} {emotion.title()}")

# Main content
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📷 Input Image")
    
    img_input = None
    
    if "Webcam" in input_mode:
        img_file_buffer = st.camera_input("Take a picture", label_visibility="collapsed")
        if img_file_buffer is not None:
            bytes_data = img_file_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            img_input = cv2_img
    else:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            cv2_img = cv2.imdecode(file_bytes, 1)
            img_input = cv2_img
            st.image(cv2_img, channels="BGR", use_container_width=True)

with col2:
    st.markdown("### 🎯 Analysis Result")
    
    if img_input is not None:
        if st.button("🔍 Analyze Mood", use_container_width=True):
            with st.spinner("🧠 Analyzing facial expression..."):
                gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                if len(faces) == 0:
                    st.warning("😕 No face detected in the image. Please try again with a clearer photo.")
                else:
                    for i, (x, y, w, h) in enumerate(faces):
                        # Draw rectangle on image
                        cv2.rectangle(img_input, (x, y), (x+w, y+h), (102, 126, 234), 3)
                        
                        # Preprocess for model
                        img_size = model.input_shape[1]
                        roi_gray = gray[y:y+h, x:x+w]
                        roi_gray = cv2.resize(roi_gray, (img_size, img_size))
                        roi_gray = roi_gray.astype('float') / 255.0
                        roi_gray = np.expand_dims(roi_gray, axis=0)
                        roi_gray = np.expand_dims(roi_gray, axis=-1)
                        
                        # Predict
                        prediction = model.predict(roi_gray, verbose=0)
                        max_index = int(np.argmax(prediction))
                        predicted_label = labels_map.get(max_index, "unknown").lower()
                        confidence = prediction[0][max_index]
                        
                        # Get emotion style
                        style = EMOTION_STYLES.get(predicted_label, EMOTION_STYLES['neutral'])
                        
                        # Display result
                        st.markdown(f"""
                        <div class="emotion-result" style="background: {style['color']}20; border: 2px solid {style['color']};">
                            <div class="emotion-emoji">{style['emoji']}</div>
                            <div class="emotion-label" style="color: {style['color']};">{predicted_label.title()}</div>
                            <div style="font-size: 1.5rem; color: #666;">Confidence: {confidence*100:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show all emotion probabilities
                        st.markdown("#### 📊 All Probabilities")
                        for idx, prob in enumerate(prediction[0]):
                            label = labels_map.get(idx, "unknown").lower()
                            em_style = EMOTION_STYLES.get(label, EMOTION_STYLES['neutral'])
                            st.progress(float(prob), text=f"{em_style['emoji']} {label.title()}: {prob*100:.1f}%")
                    
                    # Show processed image
                    st.markdown("#### 🖼️ Processed Image")
                    st.image(img_input, channels="BGR", use_container_width=True)
    else:
        st.info("👆 Please capture or upload an image to analyze")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🎓 Capstone Project: Facial Mood Detection using CNN & Transfer Learning</p>
    <p>Built with ❤️ using TensorFlow, OpenCV, and Streamlit</p>
</div>
""", unsafe_allow_html=True)
