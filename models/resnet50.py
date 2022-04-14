# Loading necessary modules

import tensorflow as tf

# Resnet50 Model : 2D CNN

def model_resnet50():
    resnet50 = tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top = False, input_shape=(256, 256, 3))
    resnet50.trainable = False

    model_resnet101 = tf.keras.models.Sequential([
        resnet101,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(2048, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model_resnet101