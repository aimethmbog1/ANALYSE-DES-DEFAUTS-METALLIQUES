import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import json

MODEL_PATH = "../models/cnn_defect_classifier.keras"
IMG_SIZE = 128
CONFIDENCE_THRESHOLD = 0.6

ALL_CLASSES = [
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
st.set_page_config(
    page_title="ANALYSE DES DEFAUTS METALLIQUES",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
.card {
    padding: 1.2rem;
    border-radius: 12px;
    background-color: white;
    border: 1px solid #e1e5eb;
    margin-bottom: 1rem;
}

.result-ok {
    background-color: #e8f5e9;
    border-left: 6px solid #2e7d32;
}

.result-warn {
    background-color: #fff3e0;
    border-left: 6px solid #ef6c00;
}

.disabled-step {
    opacity: 0.45;
    background-color: #f0f0f0;
    border-left: 6px solid #9e9e9e;
}

.big-title {
    font-size: 1.8rem;
    font-weight: 700;
}

.section-title {
    font-size: 1.4rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# CHARGEMENT DU MODELE
@st.cache_resource
def load_cnn_model():
    try:
        model = load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error("❌ Échec du chargement du modèle")
        st.code(str(e))
        raise

with st.sidebar:
    st.header("⚙️ Paramètres")
    st.markdown("### 📌 Modèle")
    st.write(f"- Taille image : **{IMG_SIZE}×{IMG_SIZE}**")
    st.write(f"- Seuil confiance : **{CONFIDENCE_THRESHOLD:.0%}**")
    st.markdown("### 🧩 Classes exploitables")
    selected_classes = []
    for cls in ALL_CLASSES:
        if st.checkbox(cls, value=True):
            selected_classes.append(cls)
    if not selected_classes:
        st.warning("⚠️ Aucune classe sélectionnée")
    st.markdown("---")
    st.info("Application d’aide à la décision.\nValidation humaine requise.")


st.title("🧠 ANALYSE DES DEFAUTS METALLIQUES")
st.markdown(
    "Analyse automatisée de défauts métalliques par Deep Learning",
)
# TELEVERSER L'IMAGE

uploaded_file = st.file_uploader(
    "📤 Charger une image à analyser",
    type=["jpg", "jpeg", "png"],
    help="Image industrielle en niveaux de gris recommandée"
)

image_loaded = uploaded_file is not None

# TABS

tab_analysis, tab_details, tab_export = st.tabs(
    ["🔍 ANALYSE", "📊 DÉTAILS", "⬇️ EXPORT"]
)

# IMAGE CHARGEMENT
if image_loaded:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if img_gray is None:
        st.error("❌ Image invalide")
        st.stop()
        
# TAB ANALYSE

with tab_analysis:
    st.markdown("<div class='section-title'>🔍 Étape 1 — Analyse</div>", unsafe_allow_html=True)

    if not image_loaded:
        st.markdown("<div class='card disabled-step'>Chargez une image pour activer cette étape.</div>", unsafe_allow_html=True)
    else:
        col_img, col_res = st.columns(2)

        with col_img:
            st.image(img_gray,caption="Image analysée",clamp=True,channels="GRAY")
        with col_res:
            with st.spinner("Analyse en cours…"):
                img = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))
                img = img.astype("float32") / 255.0
                img = np.expand_dims(img, axis=-1)
                img = np.expand_dims(img, axis=0)

                raw_preds = model.predict(img, verbose=0)[0]

            # Filtrage par classes sélectionnées
            filtered_preds = {
                cls: raw_preds[ALL_CLASSES.index(cls)]
                for cls in selected_classes
            }

            class_name = max(filtered_preds, key=filtered_preds.get)
            confidence = filtered_preds[class_name]

            if confidence >= CONFIDENCE_THRESHOLD:
                css = "result-ok"
                label = "Défaut détecté"
                icon = "✅"
            else:
                css = "result-warn"
                label = "Résultat incertain"
                icon = "⚠️"

            st.markdown(f"""
            <div class="card {css}">
                <div class="big-title">{icon} {label}</div>
                <p><strong>Classe :</strong> {class_name}</p>
                <p><strong>Confiance :</strong> {confidence:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
    
# TAB DÉTAILS

with tab_details:
    st.markdown("<div class='section-title'>📊 Étape 2 — Détails</div>", unsafe_allow_html=True)

    if not image_loaded:
        st.markdown("<div class='card disabled-step'>Analyse requise avant d’accéder aux détails.</div>", unsafe_allow_html=True)
    else:
        sorted_preds = sorted(filtered_preds.items(), key=lambda x: x[1], reverse=True)

        st.subheader("Top prédictions")
        for cls, prob in sorted_preds[:3]:
            st.write(f"- **{cls}** : {prob:.2%}")

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(list(filtered_preds.keys()), list(filtered_preds.values()))
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probabilité")
        ax.set_title("Distribution des probabilités")
        st.pyplot(fig)
        
# TAB EXPORT
with tab_export:
    st.markdown("<div class='section-title'>⬇️ Étape 3 — Export</div>", unsafe_allow_html=True)

    if not image_loaded:
        st.markdown("<div class='card disabled-step'>Analyse requise avant export.</div>", unsafe_allow_html=True)
    else:
        result = {
            "predicted_class": str(class_name),
            "confidence": float(confidence),
            "used_classes": list(selected_classes),
            "probabilities": {
                cls: float(prob)
                for cls, prob in filtered_preds.items()
            }
        }


        st.download_button(
            "Télécharger le résultat (JSON)",
            data=json.dumps(result, indent=2),
            file_name="prediction.json",
            mime="application/json"
        )
