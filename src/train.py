import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from model import create_model
import matplotlib.pyplot as plt

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

def train(epochs=50, batch_size=32):
    # Check if data exists
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        print("Data directory is empty or missing. Please run src/capture_data.py first.")
        return

    # Data Augmentation and generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        shear_range=0.3,
        zoom_range=0.3,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # 20% for validation
    )

    print(f"Loading data from {DATA_DIR}...")
    
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(48, 48),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    validation_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(48, 48),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=True
    )

    num_classes = len(train_generator.class_indices)
    print(f"Detected classes: {train_generator.class_indices}")

    # ========== CLASS BALANCING ==========
    # Compute class weights to handle imbalanced data (e.g., 'disgust' has only 436 samples)
    class_labels = train_generator.classes
    unique_classes = np.unique(class_labels)
    
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=class_labels
    )
    class_weight_dict = dict(zip(unique_classes, class_weights))
    
    print("Class weights (to balance underrepresented classes):")
    for class_idx, weight in class_weight_dict.items():
        class_name = [k for k, v in train_generator.class_indices.items() if v == class_idx][0]
        print(f"  {class_name}: {weight:.3f}")
    # =====================================

    # Create Model
    model = create_model(num_classes=num_classes)

    # Callbacks
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(MODELS_DIR, 'best_model.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # Train with class weights
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        callbacks=[checkpoint, early_stop],
        class_weight=class_weight_dict  # <-- This balances training!
    )

    # Save final model
    model.save(os.path.join(MODELS_DIR, 'final_model.keras'))
    
    # Save class indices
    with open(os.path.join(MODELS_DIR, 'class_indices.txt'), 'w') as f:
        f.write(str(train_generator.class_indices))

    # Plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss')
    
    plt.savefig(os.path.join(MODELS_DIR, 'training_history.png'))
    print("Training complete. Model saved.")

if __name__ == "__main__":
    train()
