import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Paths
DATASET_PATH = "dataset"
categories = os.listdir(DATASET_PATH)
label_dict = {category: idx for idx, category in enumerate(categories)}

# Feature extraction using color histogram
def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return hist.flatten()

# Load dataset
X, y = [], []
for category in categories:
    path = os.path.join(DATASET_PATH, category)
    for img_file in os.listdir(path):
        try:
            img_path = os.path.join(path, img_file)
            features = extract_features(img_path)
            X.append(features)
            y.append(label_dict[category])
        except Exception as e:
            print(f"Error reading {img_file}: {e}")

X, y = np.array(X), np.array(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save model and label map
joblib.dump(model, "ewaste_model.pkl")
joblib.dump(label_dict, "label_dict.pkl")
print("Model and labels saved.")
