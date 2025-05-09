# 1. Import Libraries
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# 2. Load and Preprocess the Images
def load_images_from_folder(folder, label, size=(64, 64)):
    features, labels = [], []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, size)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            features.append(gray.flatten())  # or use HOG here
            labels.append(label)
    return features, labels

# 3. Prepare Dataset
cat_features, cat_labels = load_images_from_folder('cats_and_dogs_filtered/train/cats', 0)
dog_features, dog_labels = load_images_from_folder('cats_and_dogs_filtered/train/dogs', 1)

X = np.array(cat_features + dog_features)
y = np.array(cat_labels + dog_labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM or KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred2 = knn.predict(X_test)

# 5. Evaluate the Model
print("KNN Accuracy")
print("Accuracy:", accuracy_score(y_test, y_pred2)*100)
print("Classification Report:\n", classification_report(y_test, y_pred2, target_names=["Cat", "Dog"]))




