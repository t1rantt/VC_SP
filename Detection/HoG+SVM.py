import cv2
import glob
import numpy as np
import time
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix

# Function for sliding window
def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

# Prepare positive and negative examples
positive_images = glob.glob('./SVM/good/*') # Update the path
negative_images = glob.glob('./SVM/bad/*') # Update the path

positive_features = [hog(cv2.resize(cv2.imread(file,0), (360, 360)), orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1)) for file in positive_images]
negative_features = [hog(cv2.resize(cv2.imread(file,0), (360, 360)), orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1)) for file in negative_images]

# Combine positive and negative features
features = positive_features + negative_features
labels = [1]*len(positive_features) + [0]*len(negative_features) # Assuming 1: car, 0: not car

# Split the Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train the SVM Classifier
clf = svm.SVC()
clf.fit(X_train, y_train)

# Make Predictions and Evaluate the Classifier
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

# Print confusion matrix
print("Confusion Matrix:")
print(cm)

# Measure Execution Time
start_time = time.time()

# Detect Objects in New Images and Track Objects
windowSize = (360, 360) # Window size for the sliding window
stepSize = 180 # Step size for the sliding window, adjust as necessary

# .... The beginning of your code remains unchanged ....

path = '/home/t1rant/Desktop/VC_SP/MotorcycleChase/MotorcycleChase/img/*'
files = glob.glob(path)
files.sort() # Sorting the images
images_gray = [cv2.imread(file,0) for file in files] # Loading in grayscale for analysis
images_color = [cv2.imread(file) for file in files] # Loading in color for display

for idx, (img_gray, img_color) in enumerate(zip(images_gray, images_color)):
    isLost = 1
    img_copy = img_color.copy() # Make a copy of the color image for display purposes
    bboxes = [] # List to keep all bounding boxes and their scores
    for (x, y, window) in sliding_window(img_gray, stepSize, windowSize):
        if window.shape[0] != windowSize[1] or window.shape[1] != windowSize[0]:
            continue
        new_features = hog(window, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
        score = clf.decision_function(new_features.reshape(1, -1))
        if score > 0: # Assuming positive score means the object of interest was detected
            bboxes.append(((x, y, windowSize[0], windowSize[1]), score))
    
    # Only draw the bounding box with the highest score
    if bboxes:
        bbox, score = max(bboxes, key=lambda x: x[1]) # Get the bounding box with the maximum score
        x, y, w, h = bbox
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 0, 255), 2) # Draw a red bounding box
        isLost = 0

    print(f'frame_id: {idx}, bbox: {(x, y, x + w, y + h)}, isLost: {isLost}')
   
    cv2.imshow('Image with BBox', img_copy)
    cv2.waitKey(1) # Add delay of 1ms

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time}")

cv2.destroyAllWindows() # Close all OpenCV windows

