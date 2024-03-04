import glob
import pandas as pd
from sklearn.model_selection import train_test_split

# Define base directory for the dataset
base_dir = 'Dataset'  # Adjust this if your Dataset folder is nested differently

# Load Data
benign_train_images = glob.glob(f'{base_dir}/train/benign/*.jpg')
malignant_train_images = glob.glob(f'{base_dir}/train/malignant/*.jpg')
benign_test_images = glob.glob(f'{base_dir}/test/benign/*.jpg')
malignant_test_images = glob.glob(f'{base_dir}/test/malignant/*.jpg')

# Combine training and test image paths
all_images = benign_train_images + malignant_train_images + benign_test_images + malignant_test_images

# Create DataFrame
df = pd.DataFrame({'filepath': all_images})
df['label'] = df['filepath'].apply(lambda x: 'benign' if 'benign' in x else 'malignant')

# Convert labels to binary
df['label_bin'] = (df['label'] == 'malignant').astype(int)

# Since the dataset is already split into train and test folders, you might want to keep that separation
# If you still want to split the train data into train and validation, you can do so as follows:
# Filter out the train images for splitting
train_df = df[df['filepath'].str.contains('/train/')]

# Split the train images into train and validation sets
train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=10)

# The test images can be kept separate if needed for model evaluation
test_df = df[df['filepath'].str.contains('/test/')]

# Save DataFrames
train_df.to_csv('train_data.csv', index=False)
val_df.to_csv('val_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)  # Save the test data if you plan to use it for model evaluation
