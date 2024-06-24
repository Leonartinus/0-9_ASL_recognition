# TechVidvan hand Gesture Recognizer

# import necessary packages

import cv2
import numpy as np
import mediapipe as mp
import csv
# import tensorflow as tf
# from tensorflow.python.keras.models import load_model

def main():
    # initialize mediapipe
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mpDraw = mp.solutions.drawing_utils

    # Load the gesture recognizer model
    # model = load_model('mp_hand_gesture')

    # Load class names
    f = open('gesture.names', 'r')
    classNames = f.read().split('\n')
    f.close()
    print(classNames)


    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    coords = []
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

                landmarks.append(9) ############ edit this to change the lable

                coords.append(landmarks)
                # Predict gesture
                # prediction = model.predict([landmarks])
                # print(prediction)
                # classID = np.argmax(prediction)
                # className = classNames[classID]

        # show the prediction on the frame
        cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0,0,255), 2, cv2.LINE_AA)

        # Show the final output
        cv2.imshow("Output", frame) 

        if cv2.waitKey(1) == ord('q'):
            write_csv(coords)
            break

    # release the webcam and destroy all active windows
    cap.release()

    cv2.destroyAllWindows()


def write_csv(mylst):
    import csv

    fields = [
        "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4", "x5", "y5", 
        "x6", "y6", "x7", "y7", "x8", "y8", "x9", "y9", "x10", "y10",
        "x11", "y11", "x12", "y12", "x13", "y13", "x14", "y14", "x15", "y15", 
        "x16", "y16", "x17", "y17", "x18", "y18", "x19", "y19", "x20", "y20", 
        "x21", "y21", "lable"
    ]

    filename = "Leo9.csv" ############ change the name of the file

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(fields)
        writer.writerows(mylst)

main()