# build_model.py
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB7

def build_model():
    # EfficientNet Base Model
    pre_trained_model = EfficientNetB7(
        input_shape=(224, 224, 3),
        weights='imagenet',
        include_top=False
    )

    for layer in pre_trained_model.layers:
        layer.trainable = False

    # Model Architecture
    inputs = layers.Input(shape=(224, 224, 3))
    x = layers.Flatten()(inputs)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)

    # Compile the model
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),  # Corrected here
        optimizer='adam',
        metrics=['AUC']
    )

    return model
