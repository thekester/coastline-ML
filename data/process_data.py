import os
import numpy as np
from glob import glob
from tqdm import tqdm

def load_images_and_labels(image_dir, label_dir, limit=None):
    image_paths = sorted(glob(os.path.join(image_dir, '*.npy')))
    label_paths = sorted(glob(os.path.join(label_dir, '*.npy')))

    if limit:
        image_paths = image_paths[:limit]
        label_paths = label_paths[:limit]

    images = []
    labels = []
    
    for img_path, lbl_path in tqdm(zip(image_paths, label_paths), total=len(image_paths), desc="Loading images and labels"):
        images.append(np.load(img_path))
        labels.append(np.load(lbl_path))

    return np.array(images), np.array(labels)

def normalize_images(images):
    return images / 255.0

if __name__ == "__main__":
    train_image_dir = 'train/images'
    train_label_dir = 'train/labels'
    test_image_dir = 'test/images'
    test_label_dir = 'test/labels'

    train_images, train_labels = load_images_and_labels(train_image_dir, train_label_dir)
    test_images, test_labels = load_images_and_labels(test_image_dir, test_label_dir)

    train_images = normalize_images(train_images)
    test_images = normalize_images(test_images)

    print(f"Loaded {len(train_images)} training images and {len(train_labels)} training labels.")
    print(f"Loaded {len(test_images)} test images and {len(test_labels)} test labels.")
