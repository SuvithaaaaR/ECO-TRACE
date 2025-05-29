import cv2
import numpy as np
import joblib
import os
import glob

# Load model and labels
model = joblib.load("ewaste_model.pkl")
label_dict = joblib.load("label_dict.pkl")
reverse_labels = {v: k for k, v in label_dict.items()}

# Feature extraction function
def extract_features(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")

    # Use cv2.imdecode to handle special characters in file paths
    with open(image_path, 'rb') as f:
        file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"Unable to read the image file: {image_path}")

    img = cv2.resize(img, (64, 64))
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return hist.flatten()

# Predict
try:
    image_path = input("Enter image path to classify: ")
    features = extract_features(image_path).reshape(1, -1)
    probabilities = model.predict_proba(features)[0]
    threshold = 0.5  # Set threshold to 50%

    # Sort probabilities in descending order
    sorted_probabilities = sorted(enumerate(probabilities), key=lambda x: x[1], reverse=True)

    if sorted_probabilities[0][1] < threshold:
        print("Not an e-waste")
    else:
        print("Predicted Probabilities (Descending Order):")
        for class_index, probability in sorted_probabilities:
            class_name = reverse_labels[class_index]
            print(f"{class_name}: {probability * 100:.2f}%")
except Exception as e:
    print(f"Error: {e}")

# Predict for all images in a dataset folder
def classify_dataset(folder_path):
    image_paths = glob.glob(f"{folder_path}/*.*")  # Get all image files in the folder
    for image_path in image_paths:
        try:
            features = extract_features(image_path).reshape(1, -1)
            probabilities = model.predict_proba(features)[0]
            threshold = 0.5  # Set threshold to 50%

            # Sort probabilities in descending order
            sorted_probabilities = sorted(enumerate(probabilities), key=lambda x: x[1], reverse=True)

            if sorted_probabilities[0][1] >= threshold:
                print(f"Image: {image_path}")
                print("Predicted Probabilities (Descending Order):")
                for class_index, probability in sorted_probabilities:
                    class_name = reverse_labels[class_index]
                    print(f"{class_name}: {probability * 100:.2f}%")
                print("\n")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

# Uncomment the following line to classify all images in a folder
# classify_dataset("c:/Users/Suvitha.R/OneDrive/ドキュメント/ML AND DEEP LEARNING/modified-dataset/dataset")