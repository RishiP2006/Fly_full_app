import streamlit as st
import numpy as np
import cv2
import re
from huggingface_hub import hf_hub_download, HfApi

# Camera support
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

# Try imports
try:
    import tensorflow as tf
    keras_load_model = tf.keras.models.load_model
    _HAS_KERAS = True
except Exception:
    _HAS_KERAS = False

try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except Exception:
    _HAS_YOLO = False

try:
    import torch
    import torch.nn as nn
    from torchvision import models
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

st.set_page_config(page_title="Drosophila Gender Detection", layout="centered")
st.text("Drosophila Gender Detection")
st.text("Select a model and upload an image or use live camera.")

HF_REPO_ID = "RishiPTrial/drosophila-models"

@st.cache_data(show_spinner=False)
def list_hf_models():
    api = HfApi()
    try:
        files = api.list_repo_files(repo_id=HF_REPO_ID)
    except Exception as e:
        st.error("Error listing files in Hugging Face repo: {}".format(e))
        return []
    return [f for f in files if f.lower().endswith((".pt", ".keras", ".h5", ".pth")) and not f.startswith(".")]

@st.cache_data(show_spinner=False)
def build_models_info():
    files = list_hf_models()
    info = {}
    for fname in files:
        input_size = 224
        if "inceptionv3" in fname.lower():
            input_size = 299
        if fname.lower().endswith(".pt"):
            info[fname] = {"filename": fname, "type": "detection", "framework": "yolo"}
        elif fname.lower().endswith((".keras", ".h5")):
            info[fname] = {"filename": fname, "type": "classification", "framework": "keras", "input_size": input_size}
        elif fname.lower() == "model_final.pth":
            info[fname] = {"filename": fname, "type": "classification", "framework": "torch_custom", "input_size": input_size}
        elif fname.lower().endswith(".pth"):
            info[fname] = {"filename": fname, "type": "classification", "framework": "torch", "input_size": input_size}
    return info

MODELS_INFO = build_models_info()

for name, info in MODELS_INFO.items():
    if info["framework"] == "keras" and not _HAS_KERAS:
        st.error(f"Model '{name}' requires TensorFlow/Keras but it's not installed.")
    if info["framework"] == "yolo" and not _HAS_YOLO:
        st.error(f"Model '{name}' requires YOLO but it's not installed.")
    if "torch" in info["framework"] and not _HAS_TORCH:
        st.error(f"Model '{name}' requires PyTorch but it's not installed.")

def load_model_final_pth(local_path):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    state = torch.load(local_path, map_location=torch.device('cpu'))
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

@st.cache_resource(show_spinner=False)
def load_model_from_hf(filename, info):
    try:
        local_path = hf_hub_download(repo_id=HF_REPO_ID, filename=filename)
    except Exception as e:
        st.error(f"Error downloading {filename}: {e}")
        return None

    try:
        if info["framework"] == "torch_custom":
            return load_model_final_pth(local_path)
        if info["framework"] == "keras":
            from keras.applications.resnet import preprocess_input
            return keras_load_model(local_path, custom_objects={"preprocess_input": preprocess_input})
        if info["framework"] == "torch":
            model = torch.load(local_path, map_location=torch.device("cpu"))
            model.eval()
            return model
        if info["framework"] == "yolo":
            return YOLO(local_path)
    except Exception as e:
        st.error(f"Failed loading model {filename}: {e}")
        return None

    st.error(f"Unsupported model type for {filename}")
    return None

def preprocess_image(image, size):
    img = cv2.resize(image, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0

def classify(model, img_array):
    x = np.expand_dims(img_array, axis=0)
    try:
        if _HAS_KERAS and isinstance(model, tf.keras.Model):
            return model.predict(x)
        elif _HAS_TORCH and isinstance(model, torch.nn.Module):
            with torch.no_grad():
                x_torch = torch.tensor(x).permute(0, 3, 1, 2).float()
                return model(x_torch).numpy()
    except Exception as e:
        st.error(f"Prediction failed: {e}")
    return None

def interpret_classification(preds):
    arr = np.asarray(preds)
    if arr.ndim == 2 and arr.shape[1] == 2:
        idx = int(np.argmax(arr, axis=1)[0])
        probs = np.exp(arr - np.max(arr, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        return ["Male", "Female"][idx], float(probs[0][idx])
    if arr.ndim == 2 and arr.shape[1] == 1:
        val = float(arr[0][0])
        prob = 1 / (1 + np.exp(-val)) if val < 0 or val > 1 else val
        label = "Female" if prob >= 0.5 else "Male"
        confidence = prob if label == "Female" else 1 - prob
        return label, confidence
    return None, None

def detect_yolo(model, image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    try:
        results = model.predict(img_rgb)
    except Exception as e:
        st.error(f"YOLO inference failed: {e}")
        return []
    detections = []
    for r in results:
        for b in r.boxes:
            cls = int(b.cls[0])
            conf = float(b.conf[0])
            box = b.xyxy[0].cpu().numpy().astype(int)
            name = model.names.get(cls, str(cls))
            detections.append((name, conf, box))
    return detections

class GenderDetectionProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = None
        self.model_info = None

    def update_model(self, model, info):
        self.model = model
        self.model_info = info

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        if not self.model or not self.model_info:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        if self.model_info["type"] == "classification":
            size = self.model_info.get("input_size", 224)
            img_in = preprocess_image(img, size)
            preds = classify(self.model, img_in)
            label, conf = interpret_classification(preds)
            if label:
                cv2.putText(img, f"{label} ({conf:.1%})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            detections = detect_yolo(self.model, img)
            for name, conf, box in detections:
                x1, y1, x2, y2 = box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{name} {conf:.2f}", (x1, max(y1 - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

safe_to_real = {re.sub(r"[^\w\s.-]", "_", name): name for name in MODELS_INFO}
safe_names = list(safe_to_real.keys())

safe_choice = st.selectbox("Select model", safe_names) if safe_names else None
model_choice = safe_to_real.get(safe_choice, None)
model = load_model_from_hf(model_choice, MODELS_INFO[model_choice]) if model_choice else None

st.markdown("---")
st.subheader("📷 Upload Image")
img_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
if img_file and model is not None:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is not None:
        st.image(img[..., ::-1], caption="Uploaded image", use_column_width=True)
        info = MODELS_INFO[model_choice]
        if info["type"] == "classification":
            size = info.get("input_size", 224)
            preds = classify(model, preprocess_image(img, size))
            label, conf = interpret_classification(preds)
            if label:
                st.success(f"Prediction: {label}  Confidence: {conf:.1%}")
        else:
            detections = detect_yolo(model, img)
            disp = img.copy()
            male_count = female_count = 0
            for name, conf, box in detections:
                x1, y1, x2, y2 = box
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(disp, f"{name} {conf:.2f}", (x1, max(y1 - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if name.lower() == "male":
                    male_count += 1
                elif name.lower() == "female":
                    female_count += 1
            st.image(disp[..., ::-1], caption="Detections", use_column_width=True)
            st.info(f"Detected Males: {male_count}, Females: {female_count}")

st.markdown("---")
st.subheader("📸 Live Camera Gender Detection")
if model is not None:
    ctx = webrtc_streamer(
        key="drosophila-gender-detection",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=GenderDetectionProcessor,
        async_processing=True,
    )
    if ctx.video_processor:
        ctx.video_processor.update_model(model, MODELS_INFO[model_choice])
else:
    st.warning("Please select a model first.")
