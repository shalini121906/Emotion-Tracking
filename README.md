# Emotion_detection_with_CNN

![emotion_detection](https://github.com/datamagic2020/Emotion_detection_with_CNN/blob/main/emoition_detection.png)

### Packages need to be installed
- pip install numpy
- pip install opencv-python
- pip install keras
- pip3 install --upgrade tensorflow
- pip install pillow

### download FER2013 dataset
- from below link and put in data folder under your project directory
- https://www.kaggle.com/msambare/fer2013



###step-by-step workflow for performing video emotion tracking using an emotion detection model in a Colab environment.


## Mount Google Drive

Mount Google Drive to access the required files and dependencies. This step allows us to access files stored in Google Drive within the Colab environment.


## Unzip Emotion Detection Project

Unzip the Emotion Detection project from Google Drive to the Colab environment. This step extracts the project files from the zip archive into a specified directory.


## Install Dependencies

Install the required dependencies listed in the `requirements.txt` file. This step ensures that all necessary Python libraries and packages are installed to run the project.
!pip install -r /content/Emotion_detection_with_CNN-main/requirements.txt



## Change Directory
Navigate to the project directory.

%cd /content/Emotion_detection_with_CNN-main

## Train Emotion Detector
Execute the script TrainEmotionDetector.py to train the emotion detection model.

!python TrainEmotionDetector.py

## TrainEmotionDetector.py
Import Necessary Libraries: Import required packages such as OpenCV, Keras, and ImageDataGenerator from Keras.

Initialize Data Generators: Create image data generators for training and validation data, with rescaling to normalize pixel values.

Prepare Training Data: Define a generator for training data, specifying the directory, target image size, batch size, color mode, and class mode.

Prepare Validation Data: Similarly, define a generator for validation data using the same parameters as the training data generator.

Create Model Structure: Initialize a sequential model using Keras for building the neural network architecture.

Add Convolutional and Pooling Layers: Define the convolutional and pooling layers of the model to extract features from input images.

Add Dropout Layers: Insert dropout layers to prevent overfitting during training by randomly deactivating some neurons.

Flatten and Add Dense Layers: Flatten the output from convolutional layers and add dense layers for classification.

Configure Optimizer: Set up the optimizer for training the model, using Adam optimizer with a learning rate schedule.

Compile the Model: Compile the model with categorical crossentropy loss function and accuracy metrics.

Train the Model: Train the model using the fit_generator function, passing in the training and validation generators, along with training parameters.

Save Model Architecture: Save the model architecture to a JSON file for future use.

Save Trained Model Weights: Save the trained model weights to an HDF5 file for later deployment or evaluation.

## Load Emotion Detection Model
Load the trained emotion detection model from the saved files.

json_file = open('model/emotion_model.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

emotion_model = model_from_json(loaded_model_json)

## Video Emotion Tracking
Process the input video to track emotions.

Video Input: Load a video file using OpenCV's VideoCapture function.

Frame Processing: Iterate through each frame of the video.

Convert to Grayscale: Convert each frame to grayscale for better performance in emotion detection.

Face Detection: Use a pre-trained Haar cascade classifier to detect faces in the grayscale frame.

Emotion Detection: For each detected face, perform emotion detection using a pre-trained CNN model.

Emotion Prediction: Use the CNN model to predict the emotion present in each face.

Display Emotion: Draw bounding boxes around detected faces and label them with predicted emotions.

Write to Output Video: Write the processed frames with detected emotions to an output video file.

Release Resources: Release the video capture and video writer objects once processing is complete.

Close Windows: Close any open windows displaying the video frames.


```python

```
