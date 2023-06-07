import cv2
import numpy as np
import glob
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
# Set up the initial bounding box manually
bbox_initial = (495,343,172,109)  # replace these numbers with your initial bounding box

# Initialize the ORB detector
orb = cv2.ORB_create()

# Get the path to the video frames (images)
images_path = '/home/t1rant/Desktop/VC_SP/MotorcycleChase/MotorcycleChase/img/*.jpg'

# Get the sorted list of image files
image_files = sorted(glob.glob(images_path))

# Read the first frame and extract the template (object of interest)
frame = cv2.imread(image_files[0])
template = frame[bbox_initial[1]:bbox_initial[1]+bbox_initial[3], bbox_initial[0]:bbox_initial[0]+bbox_initial[2]]
kp1, des1 = orb.detectAndCompute(template, None)

# Initialize the matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

frame_id = 0  # initialize frame counter

# Iterate over the rest of the frames
for image_file in image_files[1:]:
    frame_id += 1  # increment frame counter
    frame = cv2.imread(image_file)

    # Compute ORB features for the current frame
    kp2, des2 = orb.detectAndCompute(frame, None)

    # Match the features from the template to those of the current frame
    matches = bf.match(des1, des2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt

    # Find homography
    H, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    isLost = 0  # default is not lost
    if H is None:  # if Homography is not found, the tracking is lost
        isLost = 1

    # Use homography to get the predicted bounding box
    height, width = template.shape[:2]
    pts = np.float32([[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, H)

    # Draw bounding box
    frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    # Print tracking info
    if H is not None:
        x, y = np.int32(dst[0][0])
        w = np.int32(dst[2][0][0]) - x
        h = np.int32(dst[2][0][1]) - y
        print("Frame ID: {}, xmin: {}, ymin: {}, width: {}, height: {}, isLost: {}".format(frame_id, x, y, w, h, isLost))

    # Display result
    cv2.imshow("Tracking", frame)

    # Exit if ESC key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
