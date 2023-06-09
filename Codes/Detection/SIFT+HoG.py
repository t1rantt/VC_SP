import cv2
import numpy as np

# Set up the initial bounding box manually
bbox_initial = (495, 343, 172, 109)  # replace these numbers with your initial bounding box

# Initialize the HoG descriptor
hog = cv2.HOGDescriptor()

# Initialize the SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# Read the template (object of interest)
template = cv2.imread('template.jpg')
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# Compute HoG features for the template
template_hog = hog.compute(template_gray)

# Set up the object matcher
bf = cv2.BFMatcher()

# Iterate over the frames
for frame_path in sorted(glob.glob('frames/*.jpg')):
    # Read the frame
    frame = cv2.imread(frame_path)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect regions of interest using HoG
    regions, _ = hog.detectMultiScale(frame_gray, winStride=(8, 8), padding=(4, 4), scale=1.05)

    for region in regions:
        x, y, w, h = region

        # Extract the region of interest from the frame
        roi = frame[y:y + h, x:x + w]

        # Compute SIFT features for the region of interest
        kp1, des1 = sift.detectAndCompute(template_gray, None)
        kp2, des2 = sift.detectAndCompute(roi, None)

        # Match the SIFT descriptors
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply ratio test to filter matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # Check if enough good matches are found
        if len(good_matches) > 10:
            # Extract keypoints for the template and the region
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Compute the homography matrix
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is not None:
                # Transform the template corners using the homography matrix
                template_corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                transformed_corners = cv2.perspectiveTransform(template_corners, M)

                # Draw the bounding box around the object
                frame = cv2.polylines(frame, [np.int32(transformed_corners)], True, (0, 255, 0), 2)

    # Display the frame with bounding box
    cv2.imshow('Object Detection and Matching', frame)

    # Exit if ESC key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the video capture and close windows
cv2.destroyAllWindows()
