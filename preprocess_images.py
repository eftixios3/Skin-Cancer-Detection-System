# preprocess_images.py
import tensorflow as tf

def decode_image(filepath, label=None):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, [224, 224])
    img = tf.cast(img, tf.float32) / 255.0

    # Convert label to integer using tf.py_function
    Label = tf.py_function(lambda l: 1 if l == b'malignant' else 0, [label], tf.int64)

    return img, Label
