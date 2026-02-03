"""
Script d'inférence CNN
- Charge le modèle entraîné
- Prédit la classe d'un défaut à partir d'une image grayscale
- Pipeline IDENTIQUE à l'entraînement
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "models",
    "cnn_defect_classifier.keras"
)

IMG_SIZE = 128
IMG_CHANNELS = 1  

CLASS_NAMES = [
    "crease",
    "crescent_gap",
    "inclusion",
    "oil_spot",
    "punching_hole",
    "rolled_pit",
    "silk_spot",
    "waist_folding",
    "water_spot",
    "welding_line"
]

# CHARGEMENT DU MODÈLE

model = load_model(MODEL_PATH)
print("✅ Modèle chargé avec succès")

# FONCTION DE PRÉDICTION

def predict_image(image_path: str):
    """
    Prédit la classe d'un défaut à partir d'une image grayscale

    Args:
        image_path (str): chemin vers l'image

    Returns:
        (label, confidence)
    """

    # Vérification existence fichier
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Image introuvable : {image_path}")

    # Chargement en niveaux de gris
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("❌ Impossible de charger l'image (format invalide)")

    # Redimensionnement
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Normalisation identique à ImageDataGenerator(rescale=1./255)
    img = img.astype("float32") / 255.0

    # Ajout des dimensions : (1, H, W, 1)
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    # Prédiction
    predictions = model.predict(img, verbose=0)

    class_index = int(np.argmax(predictions))
    confidence = float(predictions[0][class_index])

    return CLASS_NAMES[class_index], confidence

# TEST LOCAL

if __name__ == "__main__":

    image_path = os.path.join(
        BASE_DIR,
        "..",
        "dataset",
        "images",
        "crease",
        "img_01_425382900_00002.jpg" 
    )

    label, score = predict_image(image_path)

    print(f"🧠 Classe prédite : {label}")
    print(f"📊 Confiance : {score:.2%}")
