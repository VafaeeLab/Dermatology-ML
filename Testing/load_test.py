# Loading necessary modules
import os
import cv2
import numpy as np
import random
import sys 
sys.path.append('/content/drive/MyDrive/unsw_github')

def load_testing_data():
    lm_test=[]
    amh_test=[]
    test_name_lm=[]
    test_name_amh=[]

    for root,dirs,files in os.walk("Dataset/Test/LM"):
        for file in files:
            im = cv2.imread(os.path.join(root, file))
            im1 = cv2.resize(im, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)        
            lm_test.append(im1)
            test_name_lm.append(file)

    for root,dirs,files in os.walk("Dataset/Test/AMH"):
        for file in files:
            im = cv2.imread(os.path.join(root, file))
            im1 = cv2.resize(im, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)        
            amh_test.append(im1)
            test_name_amh.append(file)
            amh_test.append(cv2.flip(im1,1))
            test_name_amh.append(file + '1')                                                   
            amh_test.append(cv2.flip(im1,0))
            test_name_amh.append(file + '2')
            amh_test.append(cv2.flip(im1,-1))
            test_name_amh.append(file + '3')

    test_im=[]
    test_labs=[]
    test_im=lm_test+amh_test
    test_name = test_name_lm + test_name_amh

    for i in range(0,len(lm_test)):
        test_labs.append(1)
    for j in range(0,len(amh_test)):
        test_labs.append(0)

    test_fulldata_aug = list(zip(test_im, test_labs, test_name)) 
    random.shuffle(test_fulldata_aug)
    test_im, test_labs, test_name = zip(*test_fulldata_aug)

    print("Testing Data Loaded")
    print("Number of LM datapoints = ", len(lm_test))
    print("Number of AMH Data Points:", len(amh_test))
    return test_im,test_labs,test_name 


