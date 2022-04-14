# Loading necessary modules
import numpy as np
import tensorflow as tf
import os
import pandas as pd

import sys
sys.path.append('/content/drive/MyDrive/unsw_github')

# loading test dataset
from load_test import load_testing_data
test_im, test_labs, test_name  = load_testing_data()

test_y=[]
test_x=[]

test_x = np.concatenate([arr[np.newaxis] for arr in test_im])
test_y= np.asarray(test_labs)

from algorithms import get_features
df_test = get_features(test_im, test_labs, test_name) 

# Arguments: model name
import argparse
# Creating parser
parser = argparse.ArgumentParser()
parser.add_argument('--test_for_cross_val', type = bool, default = False)
parser.add_argument('--model_name', type = str, required = True)
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

done_cross_validation = args.test_for_cross_val

if done_cross_validation:
  for i in range(0,5):
    model.compile(loss=tf.keras.losses.binary_crossentropy,
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])
    model.load_weights('Training_weights/' + str(args.model_name) + '_run_' + str(i+1) + '/best_model.h5')  #change path here
    result = model.predict(test_x)
    pred = []
    for k in range(0,len(result)):
      if result[k] > 0.5 : 
        pred.append(1)
      else:
        pred.append(0)
    import csv
    fields = ["Image ID", "Labels(Actual)", "Predicted Lables", "Probability"]
    rows = []
    for j in range(len(test_y)):
      row = []
      row.append(df_test['Image_name'][j])
      row.append(test_y[j]) 
      row.append(pred[j])
      row.append(result[j])
      rows.append(row)
    file_name = str(model_name)+"_run_"+str(i+1) +".csv"

    path = r"CSV_Files" # change path to csv files

    csv_filename = os.path.join(path, file_name)
    with open(csv_filename, mode='w',newline='') as file:  #csv file names here
      writer = csv.writer(file)
      writer.writerow(fields)
      writer.writerows(rows)
    print("csv file generated")
    print("RUN_" + str(i+1))
    print("accuracy")
    print(model.evaluate(x=test_x, y=test_y))

    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    print("confusion matrix")
    print(confusion_matrix(test_y, pred))
    print("classification report")
    print(classification_report(test_y, pred))
    
  # Misclassification Report

  for i in range(0,5):
    csv_run = pd.read_csv("CSV_Files/" + str(model_name)+ "_run_" +str(i+1) +".csv")
    misclassified = []
    for index, row in csv_run.iterrows():
      if(row['Labels(Actual)'] == row['Predicted Lables']):
        misclassified.append(0)
      elif(row['Labels(Actual)'] != row['Predicted Lables']):
        if(row['Predicted Lables'] == 0):
          misclassified.append(120)
        elif(row['Predicted Lables'] == 1):
          misclassified.append(240)
    csv_run['Misclassified'] = misclassified
    csv_run.to_csv("CSV_Files/" + str(model_name) + "_Misclassified_run" + str(i+1) + ".csv", index = False)

  csv_run1 = pd.read_csv("CSV_Files/" + str(model_name) + "_Misclassified_run1.csv")
  csv_run2 = pd.read_csv("CSV_Files/" + str(model_name) + "_Misclassified_run2.csv")
  csv_run3 = pd.read_csv("CSV_Files/" + str(model_name) + "_Misclassified_run3.csv")
  csv_run4 = pd.read_csv("CSV_Files/" + str(model_name) + "_Misclassified_run4.csv")
  csv_run5 = pd.read_csv("CSV_Files/" + str(model_name) + "_Misclassified_run5.csv")
  Index = csv_run1['Image ID']
  Cols = ['run1', 'run2', 'run3', 'run4', 'run5']
  data = [csv_run1['Misclassified'], csv_run2['Misclassified'],csv_run3['Misclassified'],csv_run4['Misclassified'],csv_run5['Misclassified']]
  data = np.transpose(data)
  df = pd.DataFrame(data, index=Index, columns=Cols)
  df.to_csv("CSV_Files/heatmap.csv", index = False)
  print("csv files saved")

else:
  model.compile(loss=tf.keras.losses.binary_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])
  model.load_weights('Training_weights/' + str(args.model_name) + '/best_model.h5')  #change path here
  result = model.predict(test_x)
  pred = []
  for k in range(0,len(result)):
    if result[k] > 0.5 : 
      pred.append(1)
    else:
      pred.append(0)
  import csv
  fields = ["Image ID", "Labels(Actual)", "Predicted Lables", "Probability"]
  rows = []
  for i in range(len(test_y)):
    row = []
    row.append(df_test['Image_name'][i])
    row.append(test_y[i]) 
    row.append(pred[i])
    row.append(result[i])
    rows.append(row)
  file_name = str(model_name) + ".csv"

  path = r"CSV_Files" # change path to csv files

  csv_filename = os.path.join(path, file_name)
  with open(csv_filename, mode='w',newline='') as file:  #csv file names here
    writer = csv.writer(file)
    writer.writerow(fields)
    writer.writerows(rows)
  print("csv file generated")
  #print("RUN_" + str(i+1))
  print("accuracy")
  print(model.evaluate(x=test_x, y=test_y))

  from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

  print("confusion matrix")
  print(confusion_matrix(test_y, pred))
  print("classification report")
  print(classification_report(test_y, pred))
    
  # Misclassification Report

  csv_run = pd.read_csv("CSV_Files/" + str(model_name)+".csv")
  misclassified = []
  for index, row in csv_run.iterrows():
    if(row['Labels(Actual)'] == row['Predicted Lables']):
      misclassified.append(0)
    elif(row['Labels(Actual)'] != row['Predicted Lables']):
      if(row['Predicted Lables'] == 0):
        misclassified.append(120)
      elif(row['Predicted Lables'] == 1):
        misclassified.append(240)
  csv_run['Misclassified'] = misclassified
  csv_run.to_csv("CSV_Files/Misclassified_" + str(model_name) + ".csv", index = False)

print("csv file saved") 