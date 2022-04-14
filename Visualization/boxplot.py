# load necessary modules
import numpy as np 
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt 
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


Xtrain = df_train.drop(['Labels','Image_name'], axis=1)
ytrain = df_train.Labels
Xtest = df_test.drop(['Labels','Image_name'], axis=1)
ytest = df_test.Labels


# loading training dataset
from Training.load_train import load_training_data
all_images, all_labels, names = load_training_data()
df_train = get_features(all_images, all_labels, names)
df_test = get_features(test_im, test_labs, test_name)
# Arguments: no. of epochs, batch size, which model to use, 5 fold cross validation reqd?
import argparse
# Creating parser
parser = argparse.ArgumentParser()
parser.add_argument('--neural_network_used', type = str, required = True)
args = parser.parse_args()
neural_net = args.neural_network_used


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

accuracies = []
for i in range(0,5):
  model.compile(loss=tf.keras.losses.binary_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])
  model.load_weights('Training_weights/' + neural_net + '_run_' + str(i+1) + '/best_model.h5')  #change path here
  print("RUN_" + str(i+1))
  print("accuracy")
  accuracy = model.evaluate(x=test_x, y=test_y)
  accuracies.append(accuracy[-1]*100)


import random
import statistics

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
result_svm = []
result_rf = []
result_ada = []
result_knn = []
kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'poly']
neighbors = [1, 3, 5, 7, 9]

for i in range (0,5):
  ######## RANDOM FOREST #########
  Random = RandomForestClassifier(n_estimators=10+12*i, criterion='gini', max_depth=None, min_samples_split=2,
                               min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                               max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, 
                               random_state=None, verbose=0)
  Random.fit(Xtrain, ytrain)
  # rf=randomForest(diagnosis~.,data=mydata.train,ntree=250, mtry = 8)
  predrf= Random.predict(Xtest)
  accuracy_rf = np.mean(predrf==ytest)*100
  result_rf.append(accuracy_rf)
  
  
  ######## SUPPORT VECTOR MACHINE #########
  
  svm = SVC(gamma='scale', class_weight='balanced',probability = True, kernel = kernels[i])
  svm.fit(Xtrain, ytrain)
  preds = svm.predict(Xtest)
  accuracy_svm = np.mean(preds==ytest)*100
  result_svm.append(accuracy_svm)

  
  ######## ADABOOST #########
  ADA = AdaBoostClassifier(base_estimator=None, n_estimators=500 - 10*i, learning_rate=0.1)
  ADA.fit(Xtrain, ytrain)
  preds_ada = ADA.predict(Xtest)
  accuracy_ada = np.mean(preds_ada==ytest)*100
  result_ada.append(accuracy_ada)
  

  ########### KNN ############
  
  knn = KNeighborsClassifier(n_neighbors = neighbors[i], metric = 'minkowski', p = 2)
  knn.fit(Xtrain, ytrain)
  preds_knn = knn.predict(Xtest)
  accuracy_knn = np.mean(preds_knn==ytest)*100
  result_knn.append(accuracy_knn)
 

results_all = []
results_all.append(accuracies)
results_all.append(result_svm)
results_all.append(result_rf)
results_all.append(result_ada)
results_all.append(result_knn)
names = [neural_net, 'SVM', 'Random Forest', 'ADABoost', 'KNN']
results_all_df = pd.DataFrame(results_all)
results_all_df.insert(loc = 0, column = 'Algorithms', value = names)
results_all_df.rename(columns={0: 'Run_1', 1: 'Run_2', 2: 'Run_3',3: 'Run_4',4: 'Run_5'}, inplace=True)
results_all_df.to_csv("CSV_Files/boxplot_data.csv",index = False)
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results_all, patch_artist=True)
ax.set_xticklabels(names)
ax.set_xlabel('Models')
ax.set_ylabel('Accuracy (%)')
plt.savefig("Figures/boxplot.png")

# Violin Plot
name = ['0', neural_net, 'SVM', 'Random Forest', 'ADABoost', 'KNN' ]
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.violinplot(results_all, showmeans = True)
ax.set_xticklabels(name)
ax.set_xlabel('Models')
ax.set_ylabel('Accuracy (%)')
plt.savefig("Figures/violinplot.png")