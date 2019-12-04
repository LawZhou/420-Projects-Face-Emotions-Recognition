# CSC420 Final Project: Face Emotions Recognition

A deep learning model for human face emotions recognition. Achieved 93.3% accuracy on CK+ dataset.

# System requirement:

## Operating System: 
Windows 10

## Graphic Card: 
GeForce GTX 1080 Ti 11GB

##  Runtime Environment: 
Python 3.7 (Anaconda 2019.10)

### reauired packages:
* numpy 1.17.4
* joblib 0.14.0
* torch 1.3.1
* matplotlib 3.1.2
* scikit-learn 0.21.3
* opencv-python 4.1.1.26
* cuda 10.0

# Setup Instuctions:
* Download and install Anaconda 2019.10 - Python 3.7 : https://www.anaconda.com/distribution/#download-section 
* Download and install the latest graphic card driver: https://www.nvidia.com/Download/driverResults.aspx/155056/en-us
* Go with the mouse to the Windows Icon (lower left) and start typing "Anaconda". There should show up some matching entries. Select "Anaconda Prompt". A new command window, named "Anaconda Prompt" will open.
* In the "Anaconda Prompt", run "pip install tensorflow-gpu"
* Download and install visual studio community 2017 (2019 won't work), installed the c++ workload from visual studio
* Download and install cuda 10.0
* Go to system environment variables (search it in the windows bar) > advanced > click Environment Variables...
* Create New user variables (do not confuse with system variables )
** Variable name: CUDA_PATH, 
** Variable value: browse to the cuda directory down to the version directory (mine is C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0)
* Go back to "Anaconda Prompt". Run "pip install opencv-python"
* In "Anaconda Prompt", run "conda install -c pytorch pytorch"
* In "Anaconda Prompt", run "conda install -c pytorch torchvision"

# Testing The model
Run **evaluate.py**. The confusion matrix will show up. The current model has a 93.3% on the current testing dataset.

# How to train you own model:

1. Run **preprocess.py**. A new folder CK+ will be generated, with 7 folder corresponds to each type of emotions. 
2. Run **preprocess_ck.py**. The image is now shuffled and saved into numpy files.
3. In **feature_extraction.py**, uncomment the line 273 and 274.
4. Run **feature_extraction.py**. This is the K-Fold Cross Validation. 35 models will be trained, and the number of incorrect predictions on each training set image will be count, in order to classify images into **Easy** and **Hard**. This step will take about 10 mins.
5. Run **knn.py**. This will train a knn classifier to classify if each image is hard or easy. It is better to rerun it mulitiple times and pick a knn classifier with the max testing accuracy. 
6. A file **knn_model.joblib** will be generated. Copy it into the folder **knn_models**, rename it to **knn_model_final.joblib**.
7.  In **emotion_classifers.py**, change the value of the variable num_epochs on line 223 to a larger number, for example, 20. Run **emotion_classifers.py** to train the final model.
8. Run **evaluate.py** to evaluate the new model.
