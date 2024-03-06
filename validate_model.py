import tensorflow as tf
import pandas as pd
from preprocess_images import decode_image  # Ensure this matches your preprocessing function

def load_and_prepare_test_data(test_csv_path, batch_size=32):
    """Load test data from a CSV file and prepare it for evaluation."""
    test_df = pd.read_csv(test_csv_path)
    test_ds = tf.data.Dataset.from_tensor_slices((test_df['filepath'], test_df['label_bin']))
    test_ds = test_ds.map(decode_image).batch(batch_size)
    return test_ds

def evaluate_model(model_path, test_csv_path):
    """Load a trained model and evaluate it on the test dataset."""
    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    # Prepare the test data
    test_ds = load_and_prepare_test_data(test_csv_path)

    # Evaluate the model
    loss, accuracy = model.evaluate(test_ds)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

if __name__ == "__main__":
    # Since the files are in the same folder, you can use just the filenames
    model_path = 'model_EfficientNetB7_20240304_224022.h5'  # Adjust if your model directory name is different
    test_csv_path = 'test_data.csv'  # The name of your test data CSV file

    evaluate_model(model_path, test_csv_path)
