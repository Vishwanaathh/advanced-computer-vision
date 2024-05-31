import cv2
import mediapipe as mp
import time
#video object
cap=cv2.VideoCapture(0)
#mediapipe initializations
mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpDraw=mp.solutions.drawing_utils
#video capture
while(1):
    success,img=cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=hands.process(imgRGB)
    #draw landmarks
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)
    #show video
    
    cv2.imshow("Image",img)
    cv2.waitKey(1)
