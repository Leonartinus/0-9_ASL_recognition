# 0-9_ASL_recognition

## Description
This is a project that can recognise the American Sign Language (ASL) numbers from 0 to 9. The cv2 library is applied to identify the position of the hand in the image and the label 21 points on the frame. The coordinates of the 21 points are returned. We use pytorch framework to build the neural network and use the coordinates with corresponding number labels to train a neural network. The interpretation of the gesture can be displayed by using the neural network to predict the label given the 21 coordinates.

## Method
- cv2 library is used to recognise the hand and return the 21 coordinates
- write .py file to collect the coordinates and write them into a .csv file and label with corresponding numbers
- split the .csv dataset into training and testing parts
- build the neural network:
  - hidden layers activation function: relu
  - output layer activation function: softmax
- Train and test the model
- Make real-time prediction using the pre-trained model

