import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import cv2
import numpy as np
import joblib

# Load model and label map
model = joblib.load("ewaste_model.pkl")
label_dict = joblib.load("label_dict.pkl")
reverse_labels = {v: k for k, v in label_dict.items()}

# Feature extraction function
def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return hist.flatten()

# GUI functions
def browse_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        features = extract_features(file_path).reshape(1, -1)
        prediction = model.predict(features)[0]
        predicted_class = reverse_labels[prediction]
        result_label.config(text=f"Predicted Class: {predicted_class}")

        # Show image in GUI
        img = Image.open(file_path)
        img = img.resize((200, 200))
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk

# Build GUI
root = tk.Tk()
root.title("E-Waste Classifier")

Label(root, text="E-Waste Image Classifier", font=("Arial", 16)).pack(pady=10)

Button(root, text="Select Image", command=browse_image, width=20).pack(pady=5)

image_label = Label(root)
image_label.pack()

result_label = Label(root, text="", font=("Arial", 14))
result_label.pack(pady=10)

root.mainloop()
