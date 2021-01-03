# CSC420 Final Project: Face Emotions Recognition

A deep learning model for human face emotion recognition. 

An implementation to the paper "Facial Expression Recognition Based on Complexity Perception Classification Algorithm"


##  Runtime Environment: 
Python 3.8


# How to train you own model:

Uncomment line 47 and 48 in **main.py** and run it.

# Testing The model
Comment out line 47 and 48 in **main.py** and run it. The confusion matrix will show up with accuracy.

# Modules
1.**preprocessCK.py**. The images are now shuffled and split into training and testing.

2.**featureExtraction.py**, This is the K-Fold Cross Validation. 
m*k_folds models will be trained, 
and the number of incorrect predictions on each training set image will be count, 
in order to classify images into **Easy** and **Hard**. 

3.**knn.py**: This will train a knn classifier to classify if each image is hard or easy.
 
4.**emotionsClassifer.py** : train the final model.

5.**evaluate.py**: evaluate the new model.
