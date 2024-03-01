import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('model_EfficientNetB7_20240301_155745.h5')  # Adjust path if your model is saved elsewhere

def preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess an image."""
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)  # Expand dims to add the batch size
    img_array_preprocessed = img_array_expanded / 255.0  # Normalize the image array
    return img_array_preprocessed

def predict_and_interpret(image_path):
    """Predict the class of an image and interpret the result."""
    processed_image = preprocess_image(image_path)

    # Make a prediction
    prediction = model.predict(processed_image)

    # Interpret the prediction
    verdict = 'Malignant' if prediction[0][0] > 0.5 else 'Benign'
    accuracy_score = prediction[0][0] if verdict == 'Malignant' else 1 - prediction[0][0]

    print(f"Verdict: {verdict}")
    print(f"Confidence: {accuracy_score:.2%}")

# Loop to keep processing new images
while True:
    image_path = input("Please enter the path to your image or type 'exit' to quit: ")
    if image_path.lower() == 'exit':
        break
    try:
        predict_and_interpret(image_path)
    except Exception as e:
        print(f"Error processing image: {e}")
