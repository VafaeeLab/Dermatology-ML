
# Loading necessary modules

import tensorflow as tf

# DenseNet169 Model: 2D CNN

def model_densenet169():
    #import tensorflow as tf
    densenet = tf.keras.applications.DenseNet169(weights='imagenet', include_top = False, input_shape=(256, 256, 3))
    densenet.trainable = False
    densenet.summary()
    model_dense = tf.keras.models.Sequential([
        densenet,
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
    return model_dense