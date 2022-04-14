# load necessary modules
import numpy as np 
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt 
import sys 
import os
import cv2
sys.path.append('/content/drive/MyDrive/unsw_github')
import tensorflow.keras.backend as K
from skimage.transform import resize
import matplotlib.pyplot as plt


LAYER_NAME = 'conv5_block32_2_conv'   
INDEX = 0
model = tf.keras.applications.DenseNet169(weights='imagenet', include_top = False, input_shape=(256, 256, 3))
grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(LAYER_NAME).output, model.output])

# Heatmap for LM 

for root,dirs,files in os.walk("/content/drive/MyDrive/Katana/Test/Test/LM"):
    for file in files:
        im_path = os.path.join(root, file)
        img = tf.keras.preprocessing.image.load_img(im_path, target_size=(256, 256))
        img = tf.keras.preprocessing.image.img_to_array(img)
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(np.array([img]))
            loss = predictions[:, INDEX]
            output = conv_outputs[0]
        grads = tape.gradient(loss, conv_outputs)[0]

        gate_f = tf.cast(output > 0, 'float32')
        gate_r = tf.cast(grads > 0, 'float32')
        guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads

        weights = tf.reduce_mean(guided_grads, axis=(0, 1))

        cam = np.ones(output.shape[0: 2], dtype = np.float32)

        for i, w in enumerate(weights):
            cam += w * output[:, :, i]

        cam = cv2.resize(cam.numpy(), (256, 256))
        cam = np.maximum(cam, 0)
        heatmap = (cam - cam.min()) / (cam.max() - cam.min())

        cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)

    
        cv2.imwrite('HeatMaps/'+'heatmap_'+'LM_'+ str(file)+'.png', cam)


# Heatmap for AMH

for root,dirs,files in os.walk("/content/drive/MyDrive/Katana/Test/Test/AMH"):
    for file in files:
        im_path = os.path.join(root, file)
        img = tf.keras.preprocessing.image.load_img(im_path, target_size=(256, 256))
        img = tf.keras.preprocessing.image.img_to_array(img)
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(np.array([img]))
            loss = predictions[:, INDEX]
            output = conv_outputs[0]
        grads = tape.gradient(loss, conv_outputs)[0]

        gate_f = tf.cast(output > 0, 'float32')
        gate_r = tf.cast(grads > 0, 'float32')
        guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads

        weights = tf.reduce_mean(guided_grads, axis=(0, 1))

        cam = np.ones(output.shape[0: 2], dtype = np.float32)

        for i, w in enumerate(weights):
            cam += w * output[:, :, i]

        cam = cv2.resize(cam.numpy(), (256, 256))
        cam = np.maximum(cam, 0)
        heatmap = (cam - cam.min()) / (cam.max() - cam.min())

        cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)

        cv2.imwrite('HeatMaps/'+'heatmap_'+'AMH_'+ str(file)+'.png', cam)


