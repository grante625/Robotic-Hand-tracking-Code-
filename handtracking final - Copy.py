import cv2
import mediapipe as mp
import time
import pyfirmata
import math
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

mpHands = mp.solutions.hands
hands = mpHands.Hands(False)
mpDraw = mp.solutions.drawing_utils

minHand, maxHand = 200, 475
minBar, maxBar = 400, 150
minAngle, maxAngle = 0, 270

port = "COM3"
board = pyfirmata.Arduino(port)
servoPin = board.get_pin('d:5:s') # pin 5 Arduino

pTime = 0
cTime = 0

def calculate_distance(x1, y1, x2, y2):
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            myHand = {}
            mylmList = []
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                mylmList.append([id, cx, cy])
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            x, y = mylmList[16][1], mylmList[16][2]
            x2, y2 = mylmList[0][1], mylmList[0][2]
            dis = calculate_distance(x, y, x2, y2)
        servoVal = np.interp(dis, [minHand, maxHand], [minAngle, maxAngle])
        servoPin.write(servoVal)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
