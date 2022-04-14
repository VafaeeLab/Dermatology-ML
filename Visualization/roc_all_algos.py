# load necessary modules
import numpy as np
import tensorflow as tf 
import sys 
sys.path.append('/content/drive/MyDrive/unsw_github')

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from algorithms import get_features

# loading test dataset
from Testing.load_test import load_testing_data
test_im, test_labs, test_name  = load_testing_data()

test_y=[]
test_x=[]

test_x = np.concatenate([arr[np.newaxis] for arr in test_im])
test_y= np.asarray(test_labs)

# loading training dataset
from Training.load_train import load_training_data
all_images, all_labels, names = load_training_data()
df_train = get_features(all_images, all_labels, names)
df_test = get_features(test_im, test_labs, test_name)

# Arguments: neural network used for training
import argparse
# Creating parser
parser = argparse.ArgumentParser()
parser.add_argument('--neural_network_used', type = str, required = True)
parser.add_argument('--cross_val_done', type = bool, default = False)
args = parser.parse_args()
neural_net = args.neural_network_used
cross_val = args.cross_val_done

if neural_net == 'resnet101':
    from models.resnet101 import model_resnet101
    model = model_resnet101()
elif neural_net == 'densenet169':
    from models.densenet169 import model_densenet169
    model = model_densenet169()
elif neural_net == 'resnet101':
    from models.resnet101 import model_resnet101
    model = model_resnet101()
elif neural_net == 'inceptionv3':
    from models.inceptionv3 import model_inceptionv3
    model = model_inceptionv3()


model.compile(loss=tf.keras.losses.binary_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

if cross_val:
  model.load_weights('Training_weights/' + str(args.neural_network_used) + '_run_' + '1' + '/best_model.h5')
else:
  model.load_weights('Training_weights/' + str(args.neural_network_used)  + '/best_model.h5')
result = model.predict(test_x)

Xtrain = df_train.drop(['Labels','Image_name'], axis=1)
ytrain = df_train.Labels
Xtest = df_test.drop(['Labels','Image_name'], axis=1)
ytest = df_test.Labels

#svm
from algorithms import model_svm
svm = model_svm()
svm.fit(Xtrain, ytrain)
preds = svm.predict(Xtest)
probs_svm = svm.predict_proba(Xtest)

# adaboost
from algorithms import model_ada
ada = model_ada()
ada.fit(Xtrain, ytrain)
preds = ada.predict(Xtest)
probs_ada = ada.predict_proba(Xtest)

# random forest
from algorithms import model_rf
rf = model_rf()
rf.fit(Xtrain, ytrain)
preds = rf.predict(Xtest)
probs_rf = rf.predict_proba(Xtest)


# calculate scores
model_auc = roc_auc_score(test_y, result)
svm_auc = roc_auc_score(ytest, probs_svm[:,1])
rf_auc = roc_auc_score(ytest, probs_rf[:,1])
ada_auc = roc_auc_score(ytest, probs_ada[:,1])

# summarize scores
print('Densenet: ROC AUC=%.3f' % (model_auc))
print('SVM: ROC AUC=%.3f' % (svm_auc))
print('RF: ROC AUC=%.3f' % (rf_auc))
print('Adaboost: ROC AUC=%.3f' % (ada_auc))

# calculate roc curves
model_fpr, model_tpr, _ = roc_curve(test_y, result)
svm_fpr, svm_tpr, _ = roc_curve(ytest, probs_svm[:,1])
rf_fpr, rf_tpr, _ = roc_curve(ytest, probs_rf[:,1])
ada_fpr, ada_tpr, _ = roc_curve(ytest, probs_ada[:,1])

# plot the roc curve for the model
pyplot.plot(model_fpr, model_tpr, label= neural_net)
pyplot.plot(svm_fpr, svm_tpr, marker='.', label='SVM')
pyplot.plot(rf_fpr, rf_tpr, marker='.', label='RF')
pyplot.plot(ada_fpr, ada_tpr, marker='.', label='Adaboost')

# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')

# show the legend
pyplot.legend()

# show the plot
pyplot.savefig('Figures/ROC_AUC_curve.png') #change path to save figures here
pyplot.savefig('Figures/ROC_AUC_curve.pdf') #change path to save figures here
pyplot.show()