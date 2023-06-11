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

image_files = sorted(glob.glob('/home/t1rant/Downloads/DriftCar1/img/*.jpg'))
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

# Spatial weights
y = np.linspace(-0.5, 0.5, roi[3])[:, np.newaxis]
x = np.linspace(-0.5, 0.5, roi[2])[np.newaxis, :]
weights = np.exp(-0.5*(x**2 + y**2))

kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 1e-4

for idx, image_file in enumerate(image_files[1:]):
    frame = cv2.imread(image_file, cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0, 1], roi_hist, [0, 180, 0, 256], 1)

    # Make sure ROI is within frame boundaries
    roi = (
        max(0, min(roi[0], frame.shape[1] - roi[2])),  # x
        max(0, min(roi[1], frame.shape[0] - roi[3])),  # y
        min(roi[2], frame.shape[1]),  # width
        min(roi[3], frame.shape[0])   # height
    )

    # Recreate weights for each ROI
    y = np.linspace(-0.5, 0.5, roi[3])[:, np.newaxis]
    x = np.linspace(-0.5, 0.5, roi[2])[np.newaxis, :]
    weights = np.exp(-0.5*(x**2 + y**2))

    # Apply spatial mask
    dst_roi = dst[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]].astype(np.float64)
    dst_roi *= weights
    dst_roi = dst_roi.astype(np.uint8)

    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(dst_roi)

    # Compute new ROI
    roi = (maxLoc[0] + roi[0], maxLoc[1] + roi[1], roi[2], roi[3])
    predicted = kalman.predict()
    x, y, w, h = roi
    measured = np.array([[np.float32(x)], [np.float32(y)]])
    corrected = kalman.correct(measured)
    roi = (int(corrected[0].item()), int(corrected[1].item()), w, h)  # Use the corrected ROI

    print("{},{},{},{},{},{}".format(idx, x, y, w, h, int(minVal == 0)))

    cv2.rectangle(frame, (roi[0], roi[1]), (roi[0]+roi[2], roi[1]+roi[3]), (0, 255, 0), 2)
    cv2.imshow('Manual Tracking', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# End timestamp
end_time = time.time()

# Calculate execution time
execution_time = end_time - start_time
print("Execution time: {:.2f} seconds".format(execution_time))

cv2.destroyAllWindows()
