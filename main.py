import os
import cv2
import numpy as np
from skimage.feature import hog
from skimage import color
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# --- HOG Feature Extraction ---
def extract_hog_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    gray = color.rgb2gray(image)
    features, _ = hog(gray, pixels_per_cell=(16, 16),
                      cells_per_block=(2, 2), visualize=True)
    return features

# --- Load Dataset ---
dataset_path = "dataset"
X = []
y = []

print("🔄 Loading dataset...")

for file in os.listdir(label_path):
    if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        continue
    file_path = os.path.join(label_path, file)
    features = extract_hog_features(file_path)
    if features is not None:
        X.append(features)
        y.append(label)


print(f"✅ Total samples loaded: {len(X)}")

# --- Check if any data was loaded ---
if len(X) == 0:
    print("❌ No data found. Make sure 'dataset/' contains images in label folders.")
    exit()

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# --- Train SVM Model ---
print("🚀 Training SVM model...")
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# --- Predict and Evaluate ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Model Accuracy: {accuracy * 100:.2f}%")

# --- Save Model ---
joblib.dump(model, "svm_blood_model.pkl")
print("💾 Model saved as 'svm_blood_model.pkl'")
