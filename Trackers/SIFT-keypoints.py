import cv2
import numpy as np
import glob
import time

# Function to select the region of interest
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

# Set up the files
image_files = sorted(glob.glob('/home/t1rant/Desktop/VC_SP/MotorcycleChase/MotorcycleChase/img/*.jpg'))

# Initialize the ROI
refPt = []
cropping = False

# Initialize the SIFT detector with limited keypoints
sift = cv2.xfeatures2d.SIFT_create(nfeatures=100)

# Initialize the BFMatcher (Brute Force Matcher)
bf = cv2.BFMatcher()

# Read the first frame
image = cv2.imread(image_files[0])
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", select_roi)

# Select ROI
while True:
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("r"):
        image = clone.copy()
    elif key == ord("c"):
        break

cv2.destroyAllWindows()
roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
old_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

# Compute SIFT features for the first frame (ROI)
kp1, des1 = sift.detectAndCompute(old_gray, None)

start_time = time.time()
for idx, image_file in enumerate(image_files[1:]):
    frame = cv2.imread(image_file)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute SIFT features for the current frame
    kp2, des2 = sift.detectAndCompute(gray, None)

    # Match descriptors
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test to find the good matches
    good = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    if len(good) >= 4:   # Check for minimum good matches
        src_pts = np.float32([ kp1[m[0].queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m[0].trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is not None:
            h,w = old_gray.shape
            pts = np.float32([ [0,0], [0,h-1], [w-1,h-1], [w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts, M)
            frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
            print(f'{idx},{dst[0][0][0]},{dst[0][0][1]},{dst[2][0][0]-dst[0][0][0]},{dst[2][0][1]-dst[0][0][1]},0')
        else:
            print(f'{idx},0,0,0,0,1')  # lost track

    else:
        print(f'{idx},0,0,0,0,1')  # lost track

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# End timestamp
end_time = time.time()

# Calculate execution time
execution_time = end_time - start_time
print("Execution time: {:.2f} seconds".format(execution_time))

cv2.destroyAllWindows()
