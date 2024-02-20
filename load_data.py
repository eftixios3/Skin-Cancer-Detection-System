# load_data.py
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

# Load Data
benign_images = glob.glob('train_cancer/benign/*.jpg')
malignant_images = glob.glob('train_cancer/malignant/*.jpg')

# Create DataFrame
df = pd.DataFrame({'filepath': benign_images + malignant_images})
df['label'] = df['filepath'].apply(lambda x: 'benign' if 'benign' in x else 'malignant')

# Convert labels to binary
df['label_bin'] = (df['label'] == 'malignant').astype(int)

# Split data
train_df, val_df = train_test_split(df, test_size=0.15, random_state=10)

# Save DataFrames
train_df.to_csv('train_data.csv', index=False)
val_df.to_csv('val_data.csv', index=False)
