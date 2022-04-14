# Derm-ML
## Description

Following repository demonstrates machine learning architectures that can correctly classify lesions between LM and AIMP(elsewhere known as atypical melanocytic hyperplasia, or AMH). Overall, our methods showcase the potential for computer-aided diagnosis in dermatology, which, in conjunction with remote acquisition can expand the range of diagnostic tools in the community. 


This code is implemented using Keras and Tensorflow frameworks.

## Role Of Machine Learning In Detection Of Lung Cancer

Machine learning based lung cancer prediction models have been proposed to assist clinicians in managing incidental or screen detected indeterminate pulmonary nodules. Such systems may be able to reduce variability in nodule classification, improve decision making and ultimately reduce the number of benign nodules that are needlessly followed or worked-up. In this project, we provide an overview of the main lung cancer prediction approaches and novel methods(LZP and SME) to image cleaning and visualization proposed to date and highlight some of their relative strengths and weaknesses. We discuss some of the challenges in the development and validation of such techniques and outline the path to clinical adoption.

## Using the Dataset

1- Download the ML Study train dataset from <> link and extract both training dataset and test folders inside the dataset_histopathology.

2- There are 2 folders of different classes of cancer cells - LM and AMH.
The images originally were stacked in folders patient-wise for both LM and AMH. There are multiple Viva Stacks inside a folder and each viva stack has around 32 images (slices).
We perform LZP - Local Z Projection on the slices and convert the 3D image to a 2D slice. <br/>

<table>
  <tr>
    <td>AMH - Local Z Projection</td>
     <td>LM - Local Z Projection</td>
   
  </tr>
  <tr>
    <td><img src="https://user-images.githubusercontent.com/34694650/163324638-c12fee38-18b9-4381-af2c-ae9aff10d543.jpg" width=480 height=480></td>
    <td><img src="https://user-images.githubusercontent.com/34694650/163324774-2c89dd49-bb33-457a-8094-167f169ff6cc.jpg" width=480 height=480></td>
    
  </tr>
 </table>


The Directory Structure is as follows:<br/>
```
|LM /
|  |-- 1344\
|	|	|-- viva stack 0001\
|	|	|	|-- v00000001
|	|	|	|-- v00000002
|	|	|	|-- v00000003
... 

|  |-- 1511\
|	|	|-- viva stack 0001\
|	|	|	|-- v00000001
|	|	|	|-- v00000002
|	|	|	|-- v00000003
... 


|AMH /
|  |-- 4151\
|	|	|-- viva stack 0001\
|	|	|	|-- v00000001
|	|	|	|-- v00000002
|	|	|	|-- v00000003
... 

|  |-- 4207\
|	|	|-- viva stack 0001\
|	|	|	|-- v00000001
|	|	|	|-- v00000002
|	|	|	|-- v00000003
... 
 ```
There is a slice level classification of cells as well which classifies an image as LM/AMH. <br/>


## FOLDERS within this repo<br/>

1.	Training: Contains scripts for training and loading the training data
2.	Testing: Contains scripts for testing and loading the test data
3.	Visualization: Contains the code for visualization of the results via boxplot, gradcam analysis, ROC - AUC curve plots and evaluating the model
4.	Data set: Contains the data set -  (LZP processed) images. These LZP projections are carried out using ImageJ application 
5.	CSV Files: 
6.	Training Weights: Stores the training weights upon running the training scripts
7.	Models: Contains the architectures of different neural networks used in the classification process
8.	Figures: Stores the figures generated after running the visualization scripts of the heatmaps, box plots, ROC - AUC and other visualization figures

***Model followed (Structure)***<br/>


![Screenshot 2022-04-14 115721](https://user-images.githubusercontent.com/34694650/163326742-d87abd4f-9048-4796-8dd1-51bff61d816c.png)

## Training model on dataset -<br/>

To run training script:
Arguments:
* **cross_val**: If you wish to go with cross validation of the models, switch to true else false (by   default, it is false)
* **epochs**: specify the number of passes of the entire training dataset the machine learning algorithm has completed
* **model_name**: Specify the model to be used (List of available model names - densenet169, resnet101, resnet50, inceptionv3)
* **batch_size**: specify the batch size with this argument<br/>
 
```!python3 Training/train.py --cross_val True --epochs 10 --model_name densenet169 --batch_size 15 ```  (for 5 fold cross validation)<br/>

```!python3 Training/train.py --epochs 10 --model_name densenet169 --batch_size 15```


## Testing model on dataset -

To run the testing script:

Arguments:<br/>
* **test_for_cross_val**: If you wish to go with cross validation of the models, switch to true else false (by   default, it is false)
* **model_name**: Specify the model to be used (List of available model names - densenet169, resnet101, resnet50, inceptionv3)<br/>

```!python3 Testing/test.py --test_for_cross_val True --model_name densenet169```  (if you have performed cross validation)<br/>
```!python3 Testing/test.py --model_name densenet169```  (if not done cross validation)<br/>

## Visualization - 

To run the visualization script:<br/>
Arguments:<br/>
* **cross_val_done**: If you performed cross validation of the models, switch to true else false (by   default, it is false)<br/>
* **neural_network_used**: Specify the model which was used during training and testing (List of available model names - densenet169, resnet101, resnet50, inceptionv3)<br/>


This code snippet is for visualizing the gradcam heatmaps<br/>
```!python3 Visualization/gradcam.py ``` (generate gradcam heatmaps for densenet169)<br/>

For visualizing the roc-auc plots:<br/>
```!python3 Visualization/roc_all_algos.py --neural_network_used densenet169 --cross_val_done True ``` (for roc curves)<br/>

For generating test metrics including roc curves for individual algos <br/>
```!python3 Visualization/evaluate_model.py --algorithm_used svm  ```

* **algorithm_used**: Specify the algorithm used in the code<br/>
* **neural_network_used**: Specify the model which was used during training and testing (List of available model names - densenet169, resnet101, resnet50, inceptionv3)
* **done_cross_val**: If you performed cross validation of the models, switch to true else false<br/>

```!python3 Visualization/evaluate_model.py --algorithm_used neural_net --neural_network_used densenet169 --done_cross_val True ```  (if you have used neural network)

For visualizing boxplots and violin plots <br/>
```!python3 Visualization/boxplot.py --neural_network_used densenet169 ``` (for generating boxplot and violinplot)


