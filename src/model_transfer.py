"""
Transfer Learning Model for Facial Expression Recognition
Uses MobileNetV2 pre-trained on ImageNet as the base model.
This typically achieves 55-65% accuracy on FER2013.
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, Layer, Concatenate
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


# Custom layer to convert grayscale to RGB (avoids Lambda serialization issues)
@tf.keras.utils.register_keras_serializable()
class GrayscaleToRGB(Layer):
    """Converts a grayscale image (1 channel) to RGB (3 channels) by repeating the channel."""
    
    def __init__(self, **kwargs):
        super(GrayscaleToRGB, self).__init__(**kwargs)
    
    def call(self, inputs):
        return tf.image.grayscale_to_rgb(inputs)
    
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (3,)
    
    def get_config(self):
        return super().get_config()


def create_transfer_model(input_shape=(96, 96, 1), num_classes=7):
    """
    Creates a transfer learning model using MobileNetV2.
    
    Args:
        input_shape: Input shape (height, width, channels). We use 96x96 for efficiency.
        num_classes: Number of emotion classes (7 for FER2013).
    
    Returns:
        Compiled Keras model.
    """
    
    # Input layer for grayscale images
    inputs = Input(shape=input_shape)
    
    # Convert grayscale to RGB using our custom layer (serializable)
    x = GrayscaleToRGB()(inputs)
    
    # Load MobileNetV2 with pre-trained ImageNet weights
    # include_top=False removes the classification head
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(96, 96, 3)
    )
    
    # Freeze the base model layers (we only train the top layers initially)
    base_model.trainable = False
    
    # Pass through base model
    x = base_model(x, training=False)
    
    # Add custom classification head
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs, outputs)
    
    # Compile with a lower learning rate (important for transfer learning)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model


def unfreeze_and_finetune(model, base_model, learning_rate=0.0001):
    """
    Unfreezes the top layers of the base model for fine-tuning.
    Call this after initial training to improve accuracy further.
    """
    # Unfreeze the top 30 layers of MobileNetV2
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    # Recompile with a very low learning rate
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


if __name__ == "__main__":
    model, base_model = create_transfer_model()
    model.summary()
    print(f"\nBase model has {len(base_model.layers)} layers")
    print("Trainable variables:", len(model.trainable_variables))
