import cv2
import numpy as np
import glob
import time

# Set up the initial bounding box manually
bbox_initial = (413,346,172,107)  # replace these numbers with your initial bounding box

# Initialize the SIFT detector
sift = cv2.SIFT_create()

# Get the path to the video frames (images)
images_path = '/home/t1rant/Downloads/CarChase1/img/*.jpg'
start_time = time.time()

# Get the sorted list of image files
image_files = sorted(glob.glob(images_path))

# Read the first frame and extract the template (object of interest)
frame = cv2.imread(image_files[0])
template = frame[bbox_initial[1]:bbox_initial[1]+bbox_initial[3], bbox_initial[0]:bbox_initial[0]+bbox_initial[2]]
kp1, des1 = sift.detectAndCompute(template, None)

# Initialize the matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params,search_params)

frame_id = 0  # initialize frame counter

# Iterate over the rest of the frames
for image_file in image_files[1:]:
    frame_id += 1  # increment frame counter
    frame = cv2.imread(image_file)

    # Compute SIFT features for the current frame
    kp2, des2 = sift.detectAndCompute(frame, None)

    # Match the features from the template to those of the current frame
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)

    # Initialize to default values
    x, y, w, h = (0, 0, 0, 0)
    isLost = 1  # Assume that the tracker is lost initially

    # If there are enough good matches, find the homography
    if len(good) >= 4:
        # Extract location of good matches
        points1 = np.zeros((len(good), 2), dtype=np.float32)
        points2 = np.zeros((len(good), 2), dtype=np.float32)
        for i, match in enumerate(good):
            points1[i, :] = kp1[match.queryIdx].pt
            points2[i, :] = kp2[match.trainIdx].pt

        # Find homography
        H, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

        if H is not None:
            # Homography found, so the tracker is not lost
            isLost = 0

            # Use homography to get the predicted bounding box
            height, width = template.shape[:2]
            pts = np.float32([[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, H)

            # Draw bounding box
            frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

            # Update tracking info
            x, y = np.int32(dst[0][0])
            w = np.int32(dst[2][0][0]) - x
            h = np.int32(dst[2][0][1]) - y

    # Print tracking info for each frame
    print("{},{},{},{},{},{}".format(frame_id, x, y, w, h, isLost))

    # Display result
    cv2.imshow("Tracking", frame)

    # Exit if ESC key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# End timestamp
end_time = time.time()

# Calculate execution time
execution_time = end_time - start_time
print("Execution time: {:.2f} seconds".format(execution_time))
cv2.destroyAllWindows()
