# imports
import pandas as pd
import numpy as np
import csv, os ,re, glob
import ast
from collections import Counter

import torch
from torchvision import transforms
from PIL import Image
import h5py


combined_df = pd.read_csv('combined_df.csv')

'-------------------------------------------------------------------------------------------------------------------------'

# Define a transformation pipeline
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor()  # Converts to a tensor and scales to [0, 1]
])

def process_image(image_path):
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            return transform(img)
    except IOError:
        print(f"Error in processing image: {image_path}")
        return None

successful_indices = []

# Attempt to process each image and collect indices of successful ones
for index, row in combined_df.iterrows():
    img_tensor = process_image(row['img_path'])
    if img_tensor is not None:
        successful_indices.append(index)
        if index % 1000 == 0:  # Print progress every 1000 images
            print(f'Processed {index}/{len(combined_df)} images')

# Create a new DataFrame with only successfully processed images
combined_df = combined_df.loc[successful_indices].reset_index(drop=True)

def save_image_to_h5(image_tensor, h5file, index):
    if image_tensor is not None:
        h5file['images'][index, ...] = image_tensor.numpy()

# Save styles to an .npy file
styles = []

# Open an h5 file for writing
with h5py.File('/home/all/processed_data/image_torchtensor_1024.h5', 'w') as h5file:
    num_images = len(combined_df)
    # Create a dataset for images using chunked storage
    chunk_size = 16  # Adjust chunk size based on your memory constraints
    h5file.create_dataset('images', shape=(num_images, 3, 1024, 1024), 
                          chunks=(chunk_size, 3, 1024, 1024), dtype=np.float32)

    # Process and save each image
    for index, row in combined_df.iterrows():
        img_tensor = process_image(row['img_path'])
        if img_tensor is not None:
            save_image_to_h5(img_tensor, h5file, index)
            styles.append(row['Style'])
            if index % 100 == 0:  # Print progress every 100 images
                print(f'Processed {index}/{num_images} images')

# Convert styles list to a NumPy array
styles_np = np.array(styles)
np.save('/home/all/processed_data/styles_1024.npy', styles_np)
print("All images have been processed and saved.")
