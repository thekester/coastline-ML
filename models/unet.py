# models/unet.py
import tensorflow as tf
#https://stackoverflow.com/questions/71000250/import-tensorflow-keras-could-not-be-resolved-after-upgrading-to-tensorflow-2
keras = tf.keras
models = tf.keras.models

def unet_model(input_shape):
    inputs = keras.layers.Input(input_shape)

    # Downsampling
    c1 = keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    c1 = keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
    p1 = keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    c2 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
    p2 = keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    c3 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c3)
    p3 = keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p3)
    c4 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c4)
    p4 = keras.layers.MaxPooling2D((2, 2))(c4)

    c5 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p4)
    c5 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c5)

    # Upsampling
    u6 = keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = keras.layers.concatenate([u6, c4])
    c6 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u6)
    c6 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c6)

    u7 = keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = keras.layers.concatenate([u7, c3])
    c7 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u7)
    c7 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c7)

    u8 = keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = keras.layers.concatenate([u8, c2])
    c8 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u8)
    c8 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c8)

    u9 = keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = keras.layers.concatenate([u9, c1])
    c9 = keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u9)
    c9 = keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c9)

    outputs = keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model
