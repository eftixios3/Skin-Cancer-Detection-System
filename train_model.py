# train_model.py
import tensorflow as tf
import numpy as np  # Import NumPy
from load_data import train_df, val_df
from preprocess_images import decode_image
from build_model import build_model
from sklearn.utils.class_weight import compute_class_weight

AUTO = tf.data.experimental.AUTOTUNE

# Image Input Pipelines
train_ds = (
    tf.data.Dataset
    .from_tensor_slices((train_df['filepath'], train_df['label_bin']))
    .map(decode_image, num_parallel_calls=AUTO)
    .shuffle(buffer_size=len(train_df))
    .batch(32)
    .prefetch(AUTO)
)

val_ds = (
    tf.data.Dataset
    .from_tensor_slices((val_df['filepath'], val_df['label_bin']))
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(32)
    .prefetch(AUTO)
)

# Calculate class weights
class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=train_df['label_bin'].values)
class_weight = {0: class_weights[0], 1: class_weights[1]}

# Train the Model
model = build_model()

# For debugging, temporarily remove class_weight to see if the issue persists
history = model.fit(train_ds, validation_data=val_ds, epochs=5, verbose=1)  # Removed class_weight for debugging

# If removing class_weight resolves the issue, consider applying class weights manually within a custom training loop



