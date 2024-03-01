# preprocess_images.py modification for decode_image function
import tensorflow as tf

def decode_image(filepath, label):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, [224, 224])
    img = tf.cast(img, tf.float32) / 255.0

    # Ensure label has a defined shape
    label = tf.expand_dims(label, -1)  # Add an extra dimension to label

    return img, label
