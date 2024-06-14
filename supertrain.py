import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from data.process_data import load_images_and_labels, normalize_images
from models.unet import unet_model
import tensorflow as tf

# Disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Set TensorFlow logging level
tf.get_logger().setLevel('INFO')

# Ensure UTF-8 encoding for stdout and stderr
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Configure TensorFlow for GPU usage
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Set threading for CPU
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)

# Load and preprocess data
train_image_dir = 'data/train/images'
train_label_dir = 'data/train/labels'

print("load_images_and_labels")
train_images, train_labels = load_images_and_labels(train_image_dir, train_label_dir)
print("normalize_images")
train_images = normalize_images(train_images)

# Reshape labels to match the model output shape (None, 256, 256, 1)
train_labels = train_labels.reshape((-1, 256, 256, 1))

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Reshape validation labels as well
y_val = y_val.reshape((-1, 256, 256, 1))

# Define model
input_shape = (256, 256, 12)  # Shape of input images
model = unet_model(input_shape)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Define callbacks
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='models/coastline_segmentation_model_epoch_{epoch:02d}.keras',
    save_freq='epoch',
    save_best_only=False,
    monitor='val_loss',
    mode='auto',
    verbose=1
)
csv_logger = tf.keras.callbacks.CSVLogger('training_log.csv', append=True)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./logs')

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[checkpoint, csv_logger, early_stopping, tensorboard],
    verbose=1  # Set verbose to 1 to see detailed logs
)

# Save the final model
model.save('models/coastline_segmentation_model_final.keras')
