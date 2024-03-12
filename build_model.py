import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from datetime import datetime

def build_and_save_resnet_model(model_identifier="ResNet50"):
    # Load the ResNet50 base model, without the top layer
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze the base model

    # Add custom layers on top of the base model
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    # Create the final model
    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Build a unique model name with a timestamp
    model_name = f"model_{model_identifier}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"

    # Save the model
    model.save(model_name)
    print(f"Model saved as {model_name}")

# Call the function to build and save the model
build_and_save_resnet_model()
