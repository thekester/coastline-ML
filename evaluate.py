import numpy as np
import tensorflow as tf
from data.process_data import load_images_and_labels, normalize_images
import matplotlib.pyplot as plt

# Load and preprocess test data
test_image_dir = 'data/test/images'
test_label_dir = 'data/test/labels'

test_images, test_labels = load_images_and_labels(test_image_dir, test_label_dir)
test_images = normalize_images(test_images)

# Load the model
model = tf.keras.models.load_model('models/coastline_segmentation_model.h5')

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
        plt.imshow(images[i][:, :, :3])  # Display first 3 channels as RGB
        plt.title('Image')

        plt.subplot(1, 3, 2)
        plt.imshow(labels[i][0], cmap='gray')
        plt.title('Ground Truth')

        plt.subplot(1, 3, 3)
        plt.imshow(predictions[i][0], cmap='gray')
        plt.title('Prediction')

        plt.show()

display_predictions(test_images, test_labels, predictions)
