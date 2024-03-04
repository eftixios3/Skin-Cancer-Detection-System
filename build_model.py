import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB7
from datetime import datetime


def build_model(model_identifier="EfficientNetB7"):
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
    x = pre_trained_model(inputs, training=False)  # Ensure that the base model is running in inference mode here
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)

    # Compile the model
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        optimizer='adam',
        metrics=['AUC']
    )

    return model


# Unique identifier for the model, e.g., using the current date and time
model_name = "model_{}_{}".format("EfficientNetB7", datetime.now().strftime("%Y%m%d_%H%M%S"))

# Build and save the model with a unique name
model = build_model()
model.save('{}.h5'.format(model_name))  # Saves the model with a unique name
