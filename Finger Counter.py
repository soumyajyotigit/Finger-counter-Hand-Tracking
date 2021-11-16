# import the opencv library
import cv2
import time
import os
import HandTrackingModule as htm
import numpy as np

# define a video capture object
vid = cv2.VideoCapture(0)

folderPath = "FingerImages"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    #print(f'{folderPath}/{imPath}')
    overlayList.append(image)

print(len(overlayList))

detector = htm.handDetector(detectionCon=0.75)

tipIds = [4, 8, 12, 16, 20]

pTime=0
while True:

    # Capture the video frame
    # by frame
    success, img = vid.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    #print(lmList)
    if len(lmList) != 0:
        fingers = []
        #for thumb
        if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        #four fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        #print(fingers)
        totalFingers = fingers.count(1)
        print(totalFingers)

    #slicing
        h, w, c = overlayList[totalFingers-1].shape
        img[0:h, 0:w] = overlayList[totalFingers-1]

        #rectangle & number
        cv2.rectangle(img, (30,247),(160,420),(0,255,0),cv2.FILLED)
        cv2.putText(img,str(totalFingers),(55,395),cv2.FONT_HERSHEY_PLAIN,10,(225,0,0),25)



    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)


    # Display the resulting frame
    cv2.imshow('Image', img)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
