# loading training dataset
from load_train import load_training_data
all_images, all_labels, names = load_training_data()

import sys 
import numpy as np
import tensorflow as tf
sys.path.append('/content/drive/MyDrive/unsw_github')

# Arguments: no. of epochs, batch size, which model to use, 5 fold cross validation reqd?
import argparse
# Creating parser
parser = argparse.ArgumentParser()
parser.add_argument('--cross_val', type = bool, default = False)
parser.add_argument('--epochs', type = int, required = True)
parser.add_argument('--model_name', type = str, required = True)
parser.add_argument('--batch_size', type = int, required = True)
args = parser.parse_args()

model_name = args.model_name

if model_name == 'resnet101':
    from models.resnet101 import model_resnet101
    model = model_resnet101()
elif model_name == 'densenet169':
    from models.densenet169 import model_densenet169
    model = model_densenet169()
elif model_name == 'resnet101':
    from models.resnet101 import model_resnet101
    model = model_resnet101()
elif model_name == 'inceptionv3':
    from models.inceptionv3 import model_inceptionv3
    model = model_inceptionv3()

n_epochs = args.epochs
want_cross_validation = args.cross_val
n_batch_size = args.batch_size

if want_cross_validation:
    # Performing cross validation for number of runs = 5
    for i in range(0,5):
    #splitting dataset for cross validation
        val_images = all_images[87*i: 87*(i+1)]
        train_images = all_images[:87*i] + all_images[87*(i+1):]
        val_labels = all_labels[87*i: 87*(i+1)]
        train_labels = all_labels[:87*i] + all_labels[87*(i+1):]
        train_x = np.concatenate([arr[np.newaxis] for arr in train_images])
        valid_x = np.concatenate([arr[np.newaxis] for arr in val_images])
        train_y= np.asarray(train_labels)
        valid_y=np.asarray(val_labels)

        #use your required model
        save_dir='Training_weights/' + str(args.model_name) + '_run_' + str(i+1) + '/'
        checkpoint = tf.keras.callbacks.ModelCheckpoint(save_dir + "best_model.h5", 
                    monitor='val_accuracy', verbose=1, 
                    save_best_only=True, mode='max')
        callbacks_list=[checkpoint]

        model.compile(loss=tf.keras.losses.binary_crossentropy,
                    optimizer=tf.keras.optimizers.Adam(lr = 0.001),
                    metrics=['accuracy'])

        print("run number = " + str(i+1))
        history=model.fit(train_x, train_y, batch_size=n_batch_size, epochs=n_epochs, verbose=1, callbacks=callbacks_list, validation_data =(valid_x, valid_y))
else:
    val_images = all_images[0: 87]
    train_images = all_images[87:] 
    val_labels = all_labels[:87]
    train_labels = all_labels[87:]
    train_x = np.concatenate([arr[np.newaxis] for arr in train_images])
    valid_x = np.concatenate([arr[np.newaxis] for arr in val_images])
    train_y= np.asarray(train_labels)
    valid_y=np.asarray(val_labels)

    #use your required model
    save_dir='Training_weights/' + str(args.model_name) + '/'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(save_dir + "best_model.h5", 
                monitor='val_accuracy', verbose=1, 
                save_best_only=True, mode='max')
    callbacks_list=[checkpoint]

    model.compile(loss=tf.keras.losses.binary_crossentropy,
                optimizer=tf.keras.optimizers.Adam(lr = 0.001),
                metrics=['accuracy'])
    history=model.fit(train_x, train_y, batch_size=n_batch_size, epochs=n_epochs, verbose=1, callbacks=callbacks_list, validation_data =(valid_x, valid_y))
