import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
import datetime


def build_model(input_shape=(224, 224, 3), num_classes=1):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze the base model

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def save_model_with_timestamp(model, base_name='model_ResNet50'):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{base_name}_{timestamp}.h5'  # Specify the HDF5 format
    model.save(filename)
    print(f'Model saved as {filename}.h5')
