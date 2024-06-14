import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from data.process_data import load_images_and_labels, normalize_images
import sys

# Disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Set the default encoding to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Load and preprocess test data
test_image_dir = 'data/test/images'
test_label_dir = 'data/test/labels'

# Load images and labels with the appropriate file format
test_images, test_labels = load_images_and_labels(test_image_dir, test_label_dir, file_format='tif')

# Check if images and labels are loaded correctly
if test_images.size == 0 or test_labels.size == 0:
    raise ValueError("No images or labels found. Please check the directories and ensure they contain the correct files.")

# Normalize images (scale pixel values to range [0, 1])
test_images = normalize_images(test_images)

# Ensure test_images and test_labels are numpy arrays
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Transpose test_images to match the expected input shape
test_images = np.transpose(test_images, (0, 2, 3, 1))

# Reshape test_labels to match the expected shape (None, 256, 256, 1)
if test_labels.shape[-1] == 1:
    test_labels = test_labels.transpose((0, 2, 3, 1))
else:
    test_labels = np.expand_dims(test_labels, axis=-1)

# Verify the shapes
print(f"Test images shape: {test_images.shape}")
print(f"Test labels shape: {test_labels.shape}")

# Add a batch dimension if not already present
if len(test_images.shape) == 3:
    test_images = np.expand_dims(test_images, axis=0)
if len(test_labels.shape) == 3:
    test_labels = np.expand_dims(test_labels, axis=0)

# Load the model
model = tf.keras.models.load_model('models/coastline_segmentation_model.keras')

# Evaluate the model
loss, accuracy = model.evaluate(test_images, test_labels)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Make predictions
predictions = model.predict(test_images)

# Display some predictions
def display_predictions(images, labels, predictions):
    for i in range(min(5, len(images))):
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow((images[i][:, :, :3] * 255).astype(np.uint8))  # Display first 3 channels as RGB
        plt.title('Image')

        plt.subplot(1, 3, 2)
        plt.imshow(labels[i][:, :, 0], cmap='gray')
        plt.title('Ground Truth')

        plt.subplot(1, 3, 3)
        plt.imshow(predictions[i][:, :, 0], cmap='gray')
        plt.title('Prediction')

        plt.show()

display_predictions(test_images, test_labels, predictions)
