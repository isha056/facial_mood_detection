"""
Training script with Transfer Learning for better accuracy.
Uses MobileNetV2 pre-trained on ImageNet.
Expected accuracy: 55-65% on FER2013.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from model_transfer import create_transfer_model, unfreeze_and_finetune
import matplotlib.pyplot as plt

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

def train_transfer(epochs_phase1=15, epochs_phase2=25, batch_size=32):
    """
    Two-phase training:
    Phase 1: Train only the classification head (frozen base)
    Phase 2: Fine-tune the top layers of the base model
    """
    
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        print("Data directory is empty or missing. Please run src/download_data.py first.")
        return

    # Data Augmentation - using 96x96 for transfer learning
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    print(f"Loading data from {DATA_DIR}...")
    print("Using 96x96 resolution for transfer learning (upscaled from 48x48)")
    
    # Note: target_size is now 96x96 for the pre-trained model
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(96, 96),  # Larger size for transfer learning
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    validation_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(96, 96),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=True
    )

    num_classes = len(train_generator.class_indices)
    print(f"Detected classes: {train_generator.class_indices}")

    # Compute class weights
    class_labels = train_generator.classes
    unique_classes = np.unique(class_labels)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=class_labels)
    class_weight_dict = dict(zip(unique_classes, class_weights))
    
    print("\nClass weights:")
    for idx, weight in class_weight_dict.items():
        name = [k for k, v in train_generator.class_indices.items() if v == idx][0]
        print(f"  {name}: {weight:.3f}")

    # ============ PHASE 1: Train classification head only ============
    print("\n" + "="*60)
    print("PHASE 1: Training classification head (base model frozen)")
    print("="*60 + "\n")
    
    model, base_model = create_transfer_model(
        input_shape=(96, 96, 1),
        num_classes=num_classes
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(MODELS_DIR, 'best_model.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    history1 = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs_phase1,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        callbacks=[checkpoint, early_stop],
        class_weight=class_weight_dict
    )

    # ============ PHASE 2: Fine-tune top layers of base model ============
    print("\n" + "="*60)
    print("PHASE 2: Fine-tuning top layers of MobileNetV2")
    print("="*60 + "\n")
    
    model = unfreeze_and_finetune(model, base_model, learning_rate=0.00005)
    
    # Reduced patience for fine-tuning
    early_stop_ft = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True
    )

    history2 = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs_phase2,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        callbacks=[checkpoint, early_stop_ft],
        class_weight=class_weight_dict
    )

    # Save final model
    model.save(os.path.join(MODELS_DIR, 'final_model.keras'))
    
    # Save class indices
    with open(os.path.join(MODELS_DIR, 'class_indices.txt'), 'w') as f:
        f.write(str(train_generator.class_indices))

    # Combine histories for plotting
    acc = history1.history['accuracy'] + history2.history['accuracy']
    val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    loss = history1.history['loss'] + history2.history['loss']
    val_loss = history1.history['val_loss'] + history2.history['val_loss']

    # Plot
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train Accuracy')
    plt.plot(val_acc, label='Val Accuracy')
    plt.axvline(x=len(history1.history['accuracy']), color='r', linestyle='--', label='Fine-tuning starts')
    plt.legend()
    plt.title('Accuracy (Transfer Learning)')
    plt.xlabel('Epoch')

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.axvline(x=len(history1.history['loss']), color='r', linestyle='--', label='Fine-tuning starts')
    plt.legend()
    plt.title('Loss (Transfer Learning)')
    plt.xlabel('Epoch')
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, 'training_history_transfer.png'))
    
    print("\n" + "="*60)
    print("Training complete! Model saved.")
    print(f"Best validation accuracy achieved: {max(val_acc)*100:.2f}%")
    print("="*60)

if __name__ == "__main__":
    train_transfer()
