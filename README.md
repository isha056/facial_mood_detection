# 🎭 Facial Mood Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Real-time facial emotion detection using Deep Learning and Computer Vision**

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Model](#-model-architecture) • [Results](#-results)

</div>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎯 **7 Emotions** | Detects Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise |
| 🧠 **Transfer Learning** | Uses MobileNetV2 pre-trained on ImageNet |
| 📷 **Real-time Detection** | Live webcam emotion detection |
| 🌐 **Web Application** | Beautiful Streamlit UI for easy use |
| ⚖️ **Class Balancing** | Handles imbalanced datasets effectively |
| 📊 **Confidence Scores** | Shows probability for all emotions |

---

## 🎬 Demo

### Web Application
```bash
./venv/bin/streamlit run src/streamlit_app.py
```

### Real-time Webcam
```bash
./venv/bin/python src/main.py
```

---

## 📁 Project Structure

```
facial_mood_detection/
├── 📂 data/
│   └── raw/                    # FER2013 dataset (7 emotion folders)
├── 📂 models/
│   ├── best_model.keras        # Trained model
│   ├── class_indices.txt       # Emotion class mappings
│   └── training_history.png    # Training visualization
├── 📂 src/
│   ├── capture_data.py         # Webcam data collection
│   ├── download_data.py        # Download FER2013 dataset
│   ├── model.py                # Basic CNN architecture
│   ├── model_transfer.py       # MobileNetV2 transfer learning
│   ├── train.py                # Basic CNN training
│   ├── train_transfer.py       # Transfer learning training
│   ├── main.py                 # Real-time webcam inference
│   └── streamlit_app.py        # Web application
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/isha056/facial_mood_detection.git
cd facial_mood_detection
```

### 2️⃣ Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Download Dataset
```bash
python src/download_data.py
```

---

## 🎯 Usage

### Option 1: Train the Model

**Basic CNN (faster, ~47% accuracy):**
```bash
python src/train.py
```

**Transfer Learning (recommended, ~48% accuracy):**
```bash
python src/train_transfer.py
```

### Option 2: Run Inference

**Web Application (recommended):**
```bash
streamlit run src/streamlit_app.py
```
Then open http://localhost:8501 in your browser.

**Real-time Webcam:**
```bash
python src/main.py
```
Press `q` (while webcam window is focused) to quit.

---

## 🧠 Model Architecture

### Transfer Learning Model (MobileNetV2)

```
┌─────────────────────────────────────────────────────────┐
│                    Input (96x96x1)                       │
│                      Grayscale                           │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│              GrayscaleToRGB Layer                        │
│                (Custom Layer)                            │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│            MobileNetV2 (Pre-trained)                     │
│           ImageNet weights, frozen                       │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│             GlobalAveragePooling2D                       │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│    Dense(256) → Dropout(0.5) → Dense(128) → Dropout(0.3)│
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│              Dense(7, softmax)                           │
│         Output: 7 emotion probabilities                  │
└─────────────────────────────────────────────────────────┘
```

### Training Strategy

| Phase | Description | Learning Rate | Epochs |
|-------|-------------|---------------|--------|
| **Phase 1** | Train classification head (base frozen) | 0.001 | 15 |
| **Phase 2** | Fine-tune top 30 layers of MobileNetV2 | 0.00005 | 25 |

---

## 📊 Results

### Dataset: FER2013

| Emotion | Training Samples | Class Weight |
|---------|-----------------|--------------|
| 😠 Angry | ~4,000 | 0.889 |
| 🤢 Disgust | ~436 | **4.074** |
| 😨 Fear | ~4,000 | 0.889 |
| 😊 Happy | ~7,000 | 0.889 |
| 😐 Neutral | ~4,900 | 0.887 |
| 😢 Sad | ~4,800 | 0.889 |
| 😲 Surprise | ~3,200 | 0.889 |

### Model Performance

| Model | Validation Accuracy | Training Time |
|-------|---------------------|---------------|
| Basic CNN | 47.1% | ~30 min |
| CNN + Class Balancing | 42.8% | ~30 min |
| **Transfer Learning** | **47.8%** | ~20 min |

> **Note:** FER2013 is a challenging dataset. Human accuracy is only ~65-72%!

---

## 🎨 Emotion Visualization

| Emotion | Emoji | Color |
|---------|-------|-------|
| Angry | 😠 | 🔴 Red |
| Disgust | 🤢 | 🟣 Purple |
| Fear | 😨 | 🟠 Orange |
| Happy | 😊 | 🟢 Green |
| Neutral | 😐 | ⚪ Gray |
| Sad | 😢 | 🔵 Blue |
| Surprise | 😲 | 🟡 Yellow |

---

## 🛠️ Technologies Used

<div align="center">

| Technology | Purpose |
|------------|---------|
| **TensorFlow/Keras** | Deep Learning Framework |
| **MobileNetV2** | Pre-trained CNN for Transfer Learning |
| **OpenCV** | Face Detection & Image Processing |
| **Streamlit** | Web Application Framework |
| **NumPy/Pandas** | Data Manipulation |
| **Matplotlib** | Training Visualization |
| **scikit-learn** | Class Weight Computation |

</div>

---

## 📝 Requirements

```
opencv-python>=4.5
numpy>=1.21
pandas>=1.3
matplotlib>=3.4
tensorflow>=2.10
scikit-learn>=1.0
streamlit>=1.20
datasets>=2.0
pillow>=9.0
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👩‍💻 Author

**Isha Sharma**

- GitHub: [@isha056](https://github.com/isha056)

---

## 🙏 Acknowledgments

- [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013) for facial expression data
- [MobileNetV2](https://arxiv.org/abs/1801.04381) for transfer learning architecture
- [Streamlit](https://streamlit.io/) for the amazing web app framework

---

<div align="center">

**⭐ Star this repo if you found it helpful!**

Made with ❤️ for Capstone Project

</div>