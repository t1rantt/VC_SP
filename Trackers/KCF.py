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

# Initialize a tracker object
tracker = cv2.TrackerKCF_create()

# Load the first frame
first_frame = cv2.imread(image_files[0])
first_frame = cv2.resize(first_frame, (0,0), fx=1.0, fy=1.0)
start_time = time.time()
# Initialize tracker with the first frame and the bbox
bbox = [413,346,172,107]  # you should replace these numbers with your initial bounding box
ok = tracker.init(first_frame, bbox)

for i, img_file in enumerate(image_files[1:]):
    frame = cv2.imread(img_file)
    frame = cv2.resize(frame, (0,0), fx=1.0, fy=1.0)

    # Update tracker
    ok, bbox = tracker.update(frame)

    # Draw bounding box
    if ok:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        isLost = 0
    else :
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
        isLost = 1

    # Display result
    cv2.imshow("Tracking", frame)

    # Print frameID, xmin, ymin, width, height
    xmin, ymin, w, h = bbox
    print("{},{},{},{},{},{}".format(i+1, xmin, ymin, w, h, isLost))
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
