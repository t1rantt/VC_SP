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

# Use both Hue and Saturation channels
roi_hist = cv2.calcHist([hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Create the Meanshift window
term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

for idx, image_file in enumerate(image_files[1:]):
    frame = cv2.imread(image_file, cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0, 1], roi_hist, [0, 180, 0, 256], 1)

    _, roi = cv2.meanShift(dst, tuple(map(int, roi)), term_criteria)

    x, y, w, h = roi

    # Check if the object is lost
    isLost = 1 if w*h == 0 else 0

    # Print tracking info
    print(f"{idx},{x},{y},{w},{h},{isLost}")

    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('Manual Tracking', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break
# End timestamp
end_time = time.time()

# Calculate execution time
execution_time = end_time - start_time
print("Execution time: {:.2f} seconds".format(execution_time))
cv2.destroyAllWindows()
