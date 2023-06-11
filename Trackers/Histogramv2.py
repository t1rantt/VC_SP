import cv2
import numpy as np
import glob
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time

def select_roi(event, x, y, flags, param):
    global refPt, cropping
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        cropping = False 
        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", image)

refPt = []
cropping = False

image_files = sorted(glob.glob('/home/t1rant/Downloads/CarChase1/img/*.jpg'))
start_time = time.time()
image = cv2.imread(image_files[0], cv2.IMREAD_COLOR)
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", select_roi)

while True:
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("r"):
        image = clone.copy()
    elif key == ord("c"):
        break

cv2.destroyAllWindows()

roi = (refPt[0][0], refPt[0][1], refPt[1][0]-refPt[0][0], refPt[1][1]-refPt[0][1])  # x, y, w, h

hsv_roi = cv2.cvtColor(clone[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]], cv2.COLOR_BGR2HSV)
roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# threshold for considering object as lost
lost_threshold = 50

# for every other image
for idx, image_file in enumerate(image_files[1:]):
    frame = cv2.imread(image_file, cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(dst)

    isLost = maxVal < lost_threshold

    roi = (maxLoc[0], maxLoc[1], roi[2], roi[3])  # update the new roi

    print(f'{idx},{roi[0]},{roi[1]},{roi[2]},{roi[3]},{isLost}')

    cv2.rectangle(frame, (roi[0], roi[1]), (roi[0]+roi[2], roi[1]+roi[3]), (0, 255, 0), 2)  # draw a bounding box around the tracked object
    cv2.imshow('Manual Tracking', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' key to exit
        break
# End timestamp
end_time = time.time()

# Calculate execution time
execution_time = end_time - start_time
print("Execution time: {:.2f} seconds".format(execution_time))
cv2.destroyAllWindows()
