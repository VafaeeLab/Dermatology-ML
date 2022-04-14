# load necessary modules
import numpy as np
import tensorflow as tf 
import sys 
sys.path.append('/content/drive/MyDrive/unsw_github')

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
Xtrain = df_train.drop(['Labels','Image_name'], axis=1)
ytrain = df_train.Labels
Xtest = df_test.drop(['Labels','Image_name'], axis=1)
ytest = df_test.Labels

# Arguments: neural network used for training
import argparse
# Creating parser
parser = argparse.ArgumentParser()
parser.add_argument('--algorithm_used', type = str, required = True)
parser.add_argument('--neural_network_used', type = str, required = False)
parser.add_argument('--done_cross_val', type = bool, default = False)
args = parser.parse_args()
algo = args.algorithm_used
neural_net = args.neural_network_used
cross_val = args.done_cross_val

if algo == 'neural_net':
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
    if cross_val == True:
        model.load_weights('Training_weights/' + str(args.neural_network_used) + '_run_' + '1' + '/best_model.h5')
    else:
        model.load_weights('Training_weights/' + str(args.neural_network_used) + '/best_model.h5')
#model.load_weights('Training_weights/' + str(args.neural_network_used)  + '/best_model.h5')
    result = model.predict(test_x)


 
    model_auc = roc_auc_score(test_y, result)
    print('Model: ROC AUC=%.3f' % (model_auc))
    model_fpr, model_tpr, _ = roc_curve(test_y, result)
    pyplot.plot(model_fpr, model_tpr, label= neural_net)
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')

# show the legend
    pyplot.legend()

# show the plot
    pyplot.savefig('Figures/' + neural_net + '_ROC_AUC_curve.png') #change path to save figures here
    pyplot.savefig('Figures/' + neural_net + '_ROC_AUC_curve.pdf') #change path to save figures here
    pyplot.show()

    print("accuracy")
    print(model.evaluate(x=test_x, y=test_y))

    result = model.predict(test_x)
    pred = []
    for k in range(0,len(result)):
        if result[k] > 0.5 : 
            pred.append(1)
        else:
            pred.append(0)

    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    print("confusion matrix")
    print(confusion_matrix(test_y, pred))
    print("classification report")
    print(classification_report(test_y, pred))

elif algo == 'svm':
    from algorithms import model_svm
    svm = model_svm()
    svm.fit(Xtrain, ytrain)
    preds = svm.predict(Xtest)
    probs_svm = svm.predict_proba(Xtest)

    svm_auc = roc_auc_score(ytest, probs_svm[:,1])
    print('SVM: ROC AUC=%.3f' % (svm_auc))
    svm_fpr, svm_tpr, _ = roc_curve(ytest, probs_svm[:,1])
    pyplot.plot(svm_fpr, svm_tpr, marker='.', label='SVM')
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')

# show the legend
    pyplot.legend()

# show the plot
    pyplot.savefig('Figures/svm_ROC_AUC_curve.png') #change path to save figures here
    pyplot.savefig('Figures/svm_ROC_AUC_curve.pdf') #change path to save figures here
    pyplot.show()

    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    print("confusion matrix")
    print(confusion_matrix(ytest, preds))
    print("classification report")
    print(classification_report(ytest, preds))

elif algo == 'rf':
    from algorithms import model_rf
    rf = model_rf()
    rf.fit(Xtrain, ytrain)
    preds = rf.predict(Xtest)
    probs_rf = rf.predict_proba(Xtest)

    rf_auc = roc_auc_score(ytest, probs_rf[:,1])
    print('RF: ROC AUC=%.3f' % (rf_auc))
    rf_fpr, rf_tpr, _ = roc_curve(ytest, probs_rf[:,1])
    pyplot.plot(rf_fpr, rf_tpr, marker='.', label='RF')
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')

# show the legend
    pyplot.legend()

# show the plot
    pyplot.savefig('Figures/rf_ROC_AUC_curve.png') #change path to save figures here
    pyplot.savefig('Figures/rf_ROC_AUC_curve.pdf') #change path to save figures here
    pyplot.show()

    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    print("confusion matrix")
    print(confusion_matrix(ytest, preds))
    print("classification report")
    print(classification_report(ytest, preds))

elif algo == 'ada':
    from algorithms import model_ada
    ada = model_ada()
    ada.fit(Xtrain, ytrain)
    preds = ada.predict(Xtest)
    probs_ada = ada.predict_proba(Xtest)

    ada_auc = roc_auc_score(ytest, probs_ada[:,1])
    print('Adaboost: ROC AUC=%.3f' % (ada_auc))
    ada_fpr, ada_tpr, _ = roc_curve(ytest, probs_ada[:,1])
    pyplot.plot(ada_fpr, ada_tpr, marker='.', label='Adaboost')
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')

# show the legend
    pyplot.legend()

# show the plot
    pyplot.savefig('Figures/Adaboost_ROC_AUC_curve.png') #change path to save figures here
    pyplot.savefig('Figures/Adaboost_ROC_AUC_curve.pdf') #change path to save figures here
    pyplot.show()

    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    print("confusion matrix")
    print(confusion_matrix(ytest, preds))
    print("classification report")
    print(classification_report(ytest, preds))

else:
    print("Please enter correct algorithm name from the list")