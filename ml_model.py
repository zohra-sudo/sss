import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib

# Charger les images et leur label
def load_images(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # Convertir en niveaux de gris
        if img is not None:
            img = cv2.resize(img, (64, 64))  # Redimensionner pour uniformiser
            images.append(img.flatten())  # Aplatir l'image en un vecteur
            labels.append(label)
    return images, labels

# Charger les datasets
color_images, color_labels = load_images("dataset/color", 1)  # 1 = Couleur
bw_images, bw_labels = load_images("dataset/bw", 0)  # 0 = Noir & Blanc

# Concaténer les données
X = np.array(color_images + bw_images)
y = np.array(color_labels + bw_labels)

# Séparer en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Enregistrer les données pour debug
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

