import tensorflow as tf
import pandas as pd
import numpy as np  # Ensure NumPy is imported
from sklearn.utils.class_weight import compute_class_weight
from preprocess_images import decode_image  # Ensure this script is set up for ResNet50 preprocessing
from build_model import build_model
AUTO = tf.data.experimental.AUTOTUNE

# Load data
train_df = pd.read_csv('train_data.csv')
val_df = pd.read_csv('val_data.csv')
test_df = pd.read_csv('test_data.csv')

# Dataset preparation
train_ds = tf.data.Dataset.from_tensor_slices((train_df['filepath'], train_df['label_bin'])).map(decode_image, num_parallel_calls=AUTO).shuffle(1000).batch(32).prefetch(AUTO)
val_ds = tf.data.Dataset.from_tensor_slices((val_df['filepath'], val_df['label_bin'])).map(decode_image, num_parallel_calls=AUTO).batch(32).prefetch(AUTO)
test_ds = tf.data.Dataset.from_tensor_slices((test_df['filepath'], test_df['label_bin'])).map(decode_image, num_parallel_calls=AUTO).batch(32).prefetch(AUTO)

# Class weights for handling imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(train_df['label_bin']), y=train_df['label_bin'].values)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

# Model building
model = build_model()

# Model training
history = model.fit(train_ds, validation_data=val_ds, epochs=5, class_weight=class_weight_dict, verbose=1)

# Model evaluation
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
