# TechVidvan hand Gesture Recognizer

# import necessary packages

import cv2
import numpy as np
import mediapipe as mp
import csv
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(42, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 21)
        self.output_layer = nn.Linear(21, 10)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.relu(self.layer3(x))
        x = self.output_layer(x)
        return self.softmax(x)

def main():
    # initialize mediapipe
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mpDraw = mp.solutions.drawing_utils

    # Load the gesture recognizer model
    model = torch.load('model.pt')

    # Load class names
    f = open('gesture.names', 'r')
    classNames = f.read().split('\n')
    f.close()
    print(classNames)


    # Initialize the webcam
    cap = cv2.VideoCapture(0)


    scaler = StandardScaler()
    with open('small_dataset.csv', 'r', newline='') as csvfile:
        heading = next(csvfile)
        X = list(csv.reader(csvfile))

    # Fit and transform your training data to get the scaler params
    X = [row[:-1] for row in X]
    scaler.fit(X)
    
    while True:
        # Read each frame from the webcam
        _, frame = cap.read()

        x, y, c = frame.shape

        # Flip the frame vertically
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get hand landmark prediction
        result = hands.process(framergb)

        # print(result)
        
        className = ''

        # post process the result
        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    # print(id, lm)
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)

                    landmarks.append(lmx)
                    landmarks.append(lmy)

                # Drawing landmarks on frames
                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

                landmarks = scaler.transform([landmarks])

                # Convert the input data to a PyTorch tensor
                landmarks = torch.tensor(landmarks, dtype=torch.float32)

                with torch.no_grad():
                    output = model(torch.Tensor(landmarks))

                # Get the predicted class (the one with the highest probability)
                prediction = torch.argmax(output, dim=1)

                print(prediction)
                className = classNames[prediction.item()]

        # show the prediction on the frame
        cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0,0,255), 2, cv2.LINE_AA)

        # Show the final output
        cv2.imshow("Output", frame) 

        if cv2.waitKey(1) == ord('q'):
            break

    # release the webcam and destroy all active windows
    cap.release()

    cv2.destroyAllWindows()

main()