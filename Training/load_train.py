
# Loading necessary modules
import os
import cv2
import random
import sys 
sys.path.append('/content/drive/MyDrive/unsw_github')

def load_training_data():
# Loading training data set 
  all_images_aug=[]
  all_labels_aug=[]

  lm = []
  amh = []

  names_lm = []
  names_amh = []


  for root,dirs,files in os.walk("Dataset/Train/LM"): # Path to Dataset Location
    for file in files:
        im = cv2.imread(os.path.join(root, file))
        im1 = cv2.resize(im, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
        names_lm.append(file)        
        lm.append(im1)

  # Path to Dataset Location
  for root,dirs,files in os.walk("Dataset/Train/AMH"):
    for file in files:
        im = cv2.imread(os.path.join(root, file))
        im1 = cv2.resize(im, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
        img1 = cv2.flip(im1,1)
        img2 = cv2.flip(im1,0)
        img3 = cv2.flip(im1,-1)
        amh.append(im1)
        names_amh.append(file)
        amh.append(cv2.flip(im1,1))
        names_amh.append(file + '1')                                                   
        amh.append(cv2.flip(im1,0))
        names_amh.append(file + '2')
        amh.append(cv2.flip(im1,-1))
        names_amh.append(file + '3')

  all_images_aug = lm + amh
  for i in range(0,len(lm)):
    all_labels_aug.append(1)
  for j in range(0,len(amh)):
    all_labels_aug.append(0)
  names = names_lm + names_amh

  image_fulldata_aug = list(zip(all_images_aug, all_labels_aug, names)) 
  random.shuffle(image_fulldata_aug)
  all_images_aug, all_labels_aug, names = zip(*image_fulldata_aug) 
  print("Training Data Loaded")
  print("Number of LM datapoints = ", len(lm))
  print("Number of AMH Data Points:", len(amh))
  return(all_images_aug, all_labels_aug, names)
