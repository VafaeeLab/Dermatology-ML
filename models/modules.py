def load_modules_for_models():
    import keras
    from keras.layers import Conv2D, Flatten, Dense, Reshape, BatchNormalization
    from keras.layers import Dropout, Input
    from keras.models import Model, Sequential
    from tensorflow.keras.optimizers import Adam, SGD
    from keras.callbacks import ModelCheckpoint
    from keras.layers.convolutional import Conv2D, MaxPooling2D

# load_modules_for_models()