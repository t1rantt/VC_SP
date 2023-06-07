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

# Initialize a list of trackers
trackers = []

# Load the first frame
first_frame = cv2.imread(image_files[0])
first_frame = cv2.resize(first_frame, (0, 0), fx=1.0, fy=1.0)

# Define the bounding boxes for the objects to track
bboxes = cv2.selectROIs("First Frame", first_frame)
for bbox in bboxes:
    tracker = cv2.TrackerMIL_create()
    trackers.append(tracker)
    tracker.init(first_frame, tuple(bbox))

for i, img_file in enumerate(image_files[1:]):
    frame = cv2.imread(img_file)
    frame = cv2.resize(frame, (0, 0), fx=1.0, fy=1.0)

    # Update the trackers with the current frame
    for j, tracker in enumerate(trackers):
        success, box = tracker.update(frame)

        if success:
            (x, y, w, h) = [int(coord) for coord in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the frame with the bounding boxes
    cv2.imshow("Tracking", frame)

    # Print frame ID, bounding boxes
    print("Frame ID:", i)
    for j, tracker in enumerate(trackers):
        success, box = tracker.update(frame)
        if success:
            print("Object", j + 1, "xmin: {}, ymin: {}, width: {}, height: {}".format(box[0], box[1], box[2], box[3]))

    # Exit if ESC key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Close all windows
cv2.destroyAllWindows()
