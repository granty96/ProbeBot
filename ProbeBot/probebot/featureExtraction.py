import json
import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load labels from text file
labels_file = "//Users/jamesdonaldson/Desktop/ProbeBot/ProbeBot/probebot/training_data_labels.json"

labels = {}  # Dictionary to store labels

# Load labeled training data
training_data_directory = "/Users/jamesdonaldson/Desktop/ProbeBot/ProbeBot/probebot/ProbeImages"

# Read each line from the file
with open(labels_file, 'r') as file:
    for line in file:
        # Parse JSON object from the line
        data = json.loads(line)
        
        # Add key-value pair to labels dictionary
        labels.update(data)
        


# Extract HOG features and prepare labels
features = []
target = []

for filename, label in labels.items():
    image_path = os.path.join(training_data_directory, filename)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(image, (64, 128))  # Resize image to standard size for HOG
    hog_features = hog(resized_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    features.append(hog_features)
    target.append(label)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train SVM classifier
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)