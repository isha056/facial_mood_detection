# Facial Mood Detection Capstone Project

This project implements a Facial Mood Detection system using OpenCV and Deep Learning (CNNs).

## Project Structure

- `data/`: Contains raw images for training.
- `models/`: Stores trained models and training history.
- `src/`: Source code.
  - `capture_data.py`: Script to capture training data from webcam.
  - `model.py`: CNN model definition.
  - `train.py`: Script to train the model.
  - `main.py`: Real-time inference application.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Data Collection
If you don't have a dataset, use the capture script to record your own face for different emotions.
```bash
python src/capture_data.py
```
Follow the on-screen instructions to capture ~100-200 images for each emotion (Happy, Sad, Angry, Neutral, Surprise).

### 2. Training
Train the CNN model on the collected data.
```bash
python src/train.py
```
This will save the best model to `models/best_model.keras` and a plot of accuracy/loss.

### 3. Real-time Detection
Run the main application to detect moods in real-time.
```bash
python src/main.py
```

## Technology Stack
- **OpenCV**: Face detection (Haar Cascades) and image processing.
- **TensorFlow/Keras**: Deep Learning model (CNN) training and inference.
- **Python**: Core programming language.