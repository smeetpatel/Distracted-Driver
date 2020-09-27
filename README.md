# Distracted-Driver

## Introduction
Distracted driving is a growing and serious threat to road safety. Distracted driving causes approximately 1.6 million fatal accidents each year. This project leverages deep learning in computer vision to visually monitor the behaviour and attention of drivers while driving and provide the driver with real-time feedback on their behaviour. To account for real driving scnearios, we also train our model to detect on night vision images. The proposed model can precisely locate where and when the driver was distracted and provide statistical visualizations.

Application of this product extends to the following domains in addition to general safety:
1. The average consumer can use this product in their car to keep a check on their driving behaviour. Individual users can use this product to keep track of their behaviours and share with their family members and close friends.
2. Fleet management companies can use this product to assess and mitigate risk by monitoring the behaviours of their drivers. Distracted behaviours can lead to demerit points in the system and merit points for attentive driving.
3. Insurance companies can provide dynamic insurance rates to fleet management and vehicle hire companies like (Lyft/Uber), drivers with positive driving scores will get discounted insurance rates. This will lead to an increase in driver accountability and passenger safety. This directly solves the root cause of the problem.

## Technology and Dataset
Distracted Driver detection using Keras, MTCNN/DNN, OpenCV and Flask.

The dataset used for this project was utilized from kaggle. You can find the original dataset available here : https://www.kaggle.com/c/state-farm-distracted-driver-detection

This dataset consists of thousands of images showing a variety of behaviors exhibited by drivers while driving. The dataset consist of multiple categories of distracted behaviours such as: 
1. safe driving
2. texting - right
3. talking on the phone - right
4. texting - left
5. talking on the phone - left
6. operating the radio
7. drinking
8. reaching behind
9. hair and makeup
10.talking to passenger

To make this project closer to industry requirements, the model is also trained on Night vision images of drivers in the car. This section also contains a variety of low light and abnormal illuminated images that simulate real driving scenarios.

Tools Used: Google Colab, Jupyter Notebook, Eclipse

## Workflow:

1. Trained MobileNet model to recognize the distracted drivers. The model was also trained on night vision images to account for real driving scenarios.

2. Used MTCNN (Multi-task Cascade Convolutional Neural Network) to detect profile face of humans in an image.
Reference : https://github.com/ipazc/mtcnn

3. After detection of human face in an image, predicted the probabilities of the behaviour in the frame using trained model weights.

4. Deployed the model on flask to make real time predictions. (Either live camera feed or upload a video)


Files:

model.py : This class will give us the predictions of our previously trained model.

camera.py : This file implements a camera class that does the following operations: 

- Get the image stream from our input (Webcam feed or from video)
- Detect faces with MTCNN and add bounding boxes
- Rescale the images and send them to our trained deep learning model 
- get the predictions back from our trained model and add the label to each frame and return the final image stream

main.py : Lastly, our main script will create a Flask app that will render our image predictions into a web page.
