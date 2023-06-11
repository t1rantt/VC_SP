import cv2
import numpy as np
import glob
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time

# Get a list of all image file paths
image_files = sorted(glob.glob('/home/t1rant/Downloads/CarChase1/img/*.jpg'))

# Initialize a list of trackers
trackers = []

# Load the first frame
first_frame = cv2.imread(image_files[0])
first_frame = cv2.resize(first_frame, (0, 0), fx=1.0, fy=1.0)

# Define the bounding boxes for the objects to track
bboxes = cv2.selectROIs("First Frame", first_frame)
cv2.destroyAllWindows()

# Create trackers for the selected objects
for bbox in bboxes:
    tracker = cv2.TrackerMIL_create()
    trackers.append(tracker)
    tracker.init(first_frame, tuple(bbox))

# Start timestamp
start_time = time.time()

for i, img_file in enumerate(image_files[1:]):
    frame = cv2.imread(img_file)
    frame = cv2.resize(frame, (0, 0), fx=1.0, fy=1.0)

    # Update the trackers with the current frame
    for j, tracker in enumerate(trackers):
        success, box = tracker.update(frame)

        if success:
            (x, y, w, h) = [int(coord) for coord in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            isLost = 0
        else:
            isLost = 1

        # Display the frame with the bounding boxes
        cv2.imshow("Tracking", frame)

        # Print frame ID, bounding boxes, and isLost
        #print("Frame ID:", i)
        for j, tracker in enumerate(trackers):
            success, box = tracker.update(frame)
            if success:
                (x, y, w, h) = [int(coord) for coord in box]
                print("{},{},{},{},{},{}".format(i+1, x, y, w, h, isLost))

    # Exit if ESC key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# End timestamp
end_time = time.time()

# Calculate execution time
execution_time = end_time - start_time
print("Execution time: {:.2f} seconds".format(execution_time))

# Close all windows
cv2.destroyAllWindows()
