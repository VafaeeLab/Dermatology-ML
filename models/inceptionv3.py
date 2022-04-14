
# Loading necessary modules

import tensorflow as tf

# InceptionV3 Model: 2D CNN

def model_inceptionv3():
    inceptionv3 = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top = False, input_shape=(256, 256, 3))
    inceptionv3.trainable = False
    inceptionv3.summary()
    model_inceptionv3 = tf.keras.models.Sequential([
        inceptionv3,
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
    return model_inceptionv3