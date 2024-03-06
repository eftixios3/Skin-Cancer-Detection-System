import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input


def decode_image(filepath, label):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)  # Ensure RGB images
    img = tf.image.resize(img, [224, 224])  # ResNet50 expects 224x224 input

    # Preprocess the image using ResNet50's preprocessing function
    img = preprocess_input(img)

    # Ensure label has a defined shape
    label = tf.expand_dims(label, -1)

    return img, label
