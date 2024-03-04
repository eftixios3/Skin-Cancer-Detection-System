import tensorflow as tf
import numpy as np
import pandas as pd  # Import pandas for CSV file handling
from sklearn.utils.class_weight import compute_class_weight
from load_data import train_df, val_df  # Assuming these are loaded from CSVs in load_data.py
from preprocess_images import decode_image
from build_model import build_model

AUTO = tf.data.experimental.AUTOTUNE

# Prepare training dataset
train_ds = (
    tf.data.Dataset
    .from_tensor_slices((train_df['filepath'], train_df['label_bin']))
    .map(decode_image, num_parallel_calls=AUTO)
    .shuffle(buffer_size=len(train_df))
    .batch(32)
    .prefetch(AUTO)
)

# Prepare validation dataset
val_ds = (
    tf.data.Dataset
    .from_tensor_slices((val_df['filepath'], val_df['label_bin']))
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(32)
    .prefetch(AUTO)
)

# Load test data from CSV
test_df = pd.read_csv('test_data.csv')
# Prepare test dataset
test_ds = (
    tf.data.Dataset
    .from_tensor_slices((test_df['filepath'], test_df['label_bin']))
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(32)
    .prefetch(AUTO)
)

# Calculate class weights
class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=train_df['label_bin'].values)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

# Build the model
model = build_model()

# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    verbose=1,
    class_weight=class_weight_dict
)

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
