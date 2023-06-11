import cv2
import numpy as np
import glob
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
# Set up the initial bounding box manually
bbox_initial = (543,294,404,177)  # replace these numbers with your initial bounding box

# Initialize the OpenCV's MOSSE tracker
tracker = cv2.legacy.TrackerMOSSE_create()

# Get the path to the video frames (images)
images_path = '/home/t1rant/Downloads/DriftCar1/img/*.jpg'  # update the path as needed
image_files = sorted(glob.glob(images_path))

# Initialize the tracker with the first frame and bounding box
frame = cv2.imread(image_files[0])
ok = tracker.init(frame, bbox_initial)
start_time = time.time()
frame_id = 0
# Iterate over the rest of the frames
for img_file in image_files[1:]:
    frame_id += 1  # increment frame_id for each iteration
    frame = cv2.imread(img_file)

    # Update tracker
    ok, bbox = tracker.update(frame)

    # Draw bounding box
    if ok:
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, 1)
        isLost = 0
    else:
        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        isLost = 1

    # Display result
    cv2.imshow("Tracking", frame)
    print("{},{},{},{},{},{}".format(frame_id, x, y, w, h, isLost))
    # Exit if ESC key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break
# End timestamp
end_time = time.time()

# Calculate execution time
execution_time = end_time - start_time
print("Execution time: {:.2f} seconds".format(execution_time))
cv2.destroyAllWindows()
