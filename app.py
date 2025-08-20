import streamlit as st
import numpy as np
import pandas as pd
import joblib
from PIL import Image
import smtplib
from email.message import EmailMessage
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from web3 import Web3
import json
import cv2
from pyzbar.pyzbar import decode
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# ===== Streamlit Config & Styling =====
st.set_page_config(page_title="Drug Legitimacy Checker", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #f7f9fc;
    }
    h1, h2, h3 {
        color: #1a1a1a;
    }
    .css-18e3th9 {
        padding: 2rem 3rem;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0px 0px 8px rgba(0,0,0,0.05);
    }
    .stButton > button {
        background-color: #3b82f6;
        color: white;
        border: None;
        border-radius: 5px;
        padding: 0.5em 1em;
        font-size: 16px;
    }
    .stButton > button:hover {
        background-color: #2563eb;
    }
    .stMetricLabel {
        font-weight: bold;
    }
    .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; margin-bottom: 2rem;'>
    <h1 style='color: #3b82f6;'>💊 Drug Legitimacy Checker</h1>
    <p style='font-size: 18px; color: #6b7280;'>A multi-layered system to detect counterfeit drugs using AI and Blockchain</p>
</div>
""", unsafe_allow_html=True)

# ===== Blockchain Setup =====
ganache_url = "http://127.0.0.1:8545"
web3 = Web3(Web3.HTTPProvider(ganache_url))

with open("contract_address.txt", "r") as f:
    contract_address = f.read().strip()

with open("contract_abi.txt", "r") as f:
    abi = json.load(f)

contract = web3.eth.contract(address=contract_address, abi=abi)

# ===== Load ML Models & Data =====
@st.cache_resource
def load_autoencoder():
    return load_model("autoencoder_model.h5")

@st.cache_resource
def load_cnn():
    return load_model("cnn_best_model.h5")

@st.cache_resource
def load_rf():
    return joblib.load("drug_legitimacy_model.pkl")

autoencoder = load_autoencoder()
cnn_model = load_cnn()
rf_model = load_rf()

le_name = joblib.load("encoder_product_name.pkl")
le_marketer = joblib.load("encoder_marketer.pkl")
le_salt = joblib.load("encoder_salt.pkl")
le_type = joblib.load("encoder_type.pkl")
le_form = joblib.load("encoder_form.pkl")

product_master = pd.read_csv("medicine_with_batch_ids_10k.csv")
product_master["mrp"] = product_master["mrp"].astype(float)
product_master["batch_id"] = product_master["batch_id"].str.strip().str.upper()

MSE_THRESHOLD = 0.005
CNN_CONFIDENCE_THRESHOLD = 0.85
LEGITIMATE_CONFIDENCE_THRESHOLD = 0.95
CLASS_NAMES = ["Fake", "Invalid", "Legitimate"]

# ===== Email Alert =====
def send_alert_email():
    sender_email = "drugapp24@gmail.com"
    sender_password = "niigvdjvcqewtbbl"
    recipient_email = "drugapp24@gmail.com"
    msg = EmailMessage()
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = "🚨 Fake Drug Detected"
    msg.set_content("Warning: A potentially fake drug has been detected by the system.")
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)
    except Exception as e:
        st.warning(f"Email failed: {e}")

# ===== Preprocessing =====
def preprocess_autoencoder(img):
    img = img.resize((128,128)).convert("RGB")
    arr = img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def preprocess_cnn(img):
    img = img.resize((224,224)).convert("RGB")
    arr = preprocess_input(img_to_array(img))
    return np.expand_dims(arr, axis=0)

def safe_encode(encoder, value):
    return encoder.transform([value])[0] if value in encoder.classes_ else 0

# ===== Sidebar Navigation =====
st.sidebar.title("📘 Project Navigation")
mode = st.sidebar.radio("Go to:", ["Structured Info", "Image Verification", "QR Verification"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📄 About This Project")
st.sidebar.info("""Detects fake drugs using:
- Random Forest (structured data)
- CNN + Autoencoder (images)
- Blockchain (Ganache smart contract)
- QR Code batch verification
""")

# ===== Mode 1: Structured Info =====
if mode == "Structured Info":
    st.subheader("📋 Structured Drug Information")

    col1, col2 = st.columns(2)
    with col1:
        pname = st.text_input("Product Name")
        marketer = st.text_input("Marketer")
        salt = st.text_input("Salt Composition")
    with col2:
        mtype = st.selectbox("Medicine Type", le_type.classes_)
        form = st.selectbox("Product Form", le_form.classes_)
        mrp = st.number_input("MRP (₹)", min_value=0.0, step=0.5)

    batch_id = st.text_input("Batch ID (printed on pack)")

    if st.button("🔍 Predict Structured Info"):
        if not pname or not marketer or not salt or not batch_id:
            st.warning("⚠️ Please fill all fields including batch ID.")
        else:
            encoded = [
                safe_encode(le_name, pname),
                safe_encode(le_marketer, marketer),
                safe_encode(le_salt, salt),
                safe_encode(le_type, mtype),
                safe_encode(le_form, form),
                mrp
            ]
            proba = rf_model.predict_proba([encoded])[0]
            pred = np.argmax(proba)
            confidence = proba[pred]

            if pred == 1:
                match = product_master[
                    (product_master["Product Name"].str.lower() == pname.strip().lower()) &
                    (product_master["Marketer"].str.lower() == marketer.strip().lower()) &
                    (product_master["salt_composition"].str.lower() == salt.strip().lower()) &
                    (product_master["Product Form"].str.lower() == form.strip().lower()) &
                    (np.isclose(product_master["mrp"], mrp, atol=2.0)) &
                    (product_master["batch_id"] == batch_id.strip().upper())
                ]
                if not match.empty:
                    st.success(f"✅ LEGITIMATE — {confidence*100:.2f}%")
                    try:
                        product, legit = contract.functions.verifyBatch(batch_id).call()
                        if legit:
                            st.success(f"✔ Blockchain confirms batch {batch_id} is legitimate: {product}")
                        else:
                            st.error(f"❌ Blockchain says batch {batch_id} is NOT legitimate")
                    except Exception as e:
                        st.error(f"Blockchain error: {e}")
                else:
                    st.error("❌ FAKE — Info mismatch.")
                    send_alert_email()
            else:
                st.error(f"❌ FAKE — {confidence*100:.2f}%")
                send_alert_email()

# ===== Mode 2: Image Verification =====
elif mode == "Image Verification":
    st.subheader("📷 Upload Drug Package Images")
    uploaded = st.file_uploader("Upload multiple images", type=["jpg","png"], accept_multiple_files=True)
    
    if uploaded:
        for file in uploaded:
            with st.expander(f"🖼️ Analyzing: {file.name}", expanded=True):
                image = Image.open(file)
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                with col2:
                    img_ae = preprocess_autoencoder(image)
                    recon = autoencoder.predict(img_ae, verbose=0)
                    mse = np.mean((img_ae - recon)**2)
                    st.metric(label="MSE (Trust Gate)", value=f"{mse:.5f}")
                    
                    if mse > MSE_THRESHOLD:
                        st.warning("⚠️ INVALID — MSE trust gate failed")
                    else:
                        img_cnn = preprocess_cnn(image)
                        preds = cnn_model.predict(img_cnn, verbose=0)[0]
                        pred_class = np.argmax(preds)
                        confidence = preds[pred_class]
                        
                        if confidence < CNN_CONFIDENCE_THRESHOLD:
                            st.warning(f"⚠️ INVALID — CNN confidence too low ({confidence*100:.2f}%)")
                        elif pred_class == 2 and confidence < LEGITIMATE_CONFIDENCE_THRESHOLD:
                            st.warning(f"⚠️ INVALID — Legitimate class but confidence too low ({confidence*100:.2f}%)")
                        else:
                            if pred_class == 0:
                                st.error(f"❌ FAKE — {confidence*100:.2f}%")
                                send_alert_email()
                            elif pred_class == 2:
                                st.success(f"✅ LEGITIMATE — {confidence*100:.2f}%")
                            else:
                                st.warning(f"⚠️ INVALID — {confidence*100:.2f}%")

# ===== Mode 3: QR Verification =====
elif mode == "QR Verification":
    st.subheader("📦 QR Code Verification")

    with st.expander("📤 Upload QR Code Image", expanded=True):
        qr_file = st.file_uploader("Upload QR Code Image", type=["png", "jpg", "jpeg"])
        if qr_file:
            img = Image.open(qr_file).convert("RGB")
            st.image(img, caption="Uploaded QR", width=300)
            decoded = decode(img)
            if decoded:
                batch_id = decoded[0].data.decode("utf-8").strip().upper()
                st.success(f"🆔 Decoded Batch ID: `{batch_id}`")
                try:
                    product, legit = contract.functions.verifyBatch(batch_id).call()
                    if legit:
                        st.success(f"✅ Blockchain confirms `{batch_id}` is legitimate: {product}")
                    else:
                        st.error(f"❌ Blockchain says `{batch_id}` is NOT legitimate")
                except Exception as e:
                    st.error(f"Blockchain error: {e}")
                details = product_master[product_master["batch_id"] == batch_id]
                if not details.empty:
                    st.markdown("### 📎 Product Details from Blockchain")
                    st.write(details.iloc[0].to_dict())
                else:
                    st.warning("⚠️ Batch ID not found in product dataset.")
            else:
                st.warning("⚠️ Could not decode the uploaded QR image.")

    with st.expander("📷 Scan QR Code using Webcam", expanded=False):
        class QRScanner(VideoTransformerBase):
            def __init__(self):
                self.last_detected = ""
            def transform(self, frame):
                img = frame.to_ndarray(format="bgr24")
                decoded_objects = decode(img)
                for obj in decoded_objects:
                    self.last_detected = obj.data.decode("utf-8")
                    points = obj.polygon
                    if len(points) == 4:
                        pts = [(p.x, p.y) for p in points]
                        cv2.polylines(img, [np.array(pts)], isClosed=True, color=(0, 255, 0), thickness=2)
                return img

        ctx = webrtc_streamer(key="qr-scanner", video_processor_factory=QRScanner)
        if ctx.video_transformer:
            qr_data = ctx.video_transformer.last_detected.strip().upper()
            st.text(f"Last Detected QR: {qr_data}")

            if st.button("🔎 Scan QR Now"):
                if qr_data:
                    st.success(f"✅ Scanned QR: `{qr_data}`")
                    try:
                        product_name, status = contract.functions.verifyBatch(qr_data).call()
                        if status:
                            st.success(f"✔ Blockchain confirms batch {qr_data} is legitimate: {product_name}")
                        else:
                            st.error(f"❌ Blockchain says batch {qr_data} is NOT legitimate")
                    except Exception as e:
                        st.error(f"Blockchain error: {e}")

                    match = product_master[product_master["batch_id"] == qr_data]
                    if not match.empty:
                        st.markdown("### 📎 Product Details from Blockchain")
                        st.write(match.iloc[0].to_dict())
                    else:
                        st.warning("⚠️ Batch ID not found in product dataset.")
                else:
                    st.warning("⚠️ No QR detected yet. Please hold steady.")
