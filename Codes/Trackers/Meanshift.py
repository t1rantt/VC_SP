import cv2
import numpy as np
import glob
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time

# Get a list of all image file paths
image_files = sorted(glob.glob('/home/t1rant/Desktop/VC_SP/MotorcycleChase/MotorcycleChase/img/*.jpg'))

# Load the first frame
first_frame = cv2.imread(image_files[0])

# Set up the initial tracking window
# bbox is a list containing the coordinates of the bounding box
bbox = [495,343,172,109]  # you should replace these numbers with your initial bounding box
(x, y, w, h) = tuple(map(int, bbox))

# set up the Region of Interest for tracking
roi = first_frame[y:y+h, x:x+w]

# Use the HSV color space for better tracking results
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# Create a mask between the HSV bounds
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))

# Calculate histogram from the masked region
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0,180])

# Normalize the histogram
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by at least 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

for i, img_file in enumerate(image_files[1:]):
    frame = cv2.imread(img_file)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0,180], 1)
    
    # apply meanshift to get the new location
    ret, bbox = cv2.meanShift(dst, bbox, term_crit)
    
    # Determine if the object is lost or not
    isLost = 0
    if ret == False:
        isLost = 1

    # Draw it on image
    x, y, w, h = bbox
    img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255, 2)
    cv2.imshow('Meanshift Tracking', img2)

    # Print frameID, xmin, ymin, width, height, isLost
    print("Frame ID: {}, xmin: {}, ymin: {}, width: {}, height: {}, isLost: {}".format(i, x, y, w, h, isLost))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
