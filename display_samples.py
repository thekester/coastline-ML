import matplotlib.pyplot as plt
import numpy as np
from data.process_data import load_images_and_labels

# Load and preprocess data
train_image_dir = 'data/train/images'
train_label_dir = 'data/train/labels'

train_images, train_labels = load_images_and_labels(train_image_dir, train_label_dir)

# Display samples function
def display_samples(images, labels):
    for i in range(min(5, len(images))):  # Display up to the first 5 samples
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(images[i][:, :, :3])  # Display first 3 channels as RGB
        plt.title(f"Sample {i} - Image")
        
        plt.subplot(1, 2, 2)
        plt.imshow(labels[i][0], cmap='gray')  # Display the label as grayscale
        plt.title(f"Sample {i} - Label")
        
        plt.show()

# Display samples
display_samples(train_images, train_labels)
