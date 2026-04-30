import os
import sys
import types

import cv2
import numpy as np
import streamlit as st


def ensure_tensorflow_compat():
    # TensorFlow may try to import JAX for optional TFLite helpers.
    # In some deployments JAX pulls an incompatible ml_dtypes version.
    if "jax" not in sys.modules:
        jax_stub = types.ModuleType("jax")

        def _xla_computation(*args, **kwargs):
            raise NotImplementedError("JAX is not available in this environment.")

        jax_stub.xla_computation = _xla_computation
        jax_stub.monitoring = types.SimpleNamespace(
            record_event=lambda *args, **kwargs: None,
            record_event_duration_secs=lambda *args, **kwargs: None,
            record_scalar=lambda *args, **kwargs: None,
        )
        sys.modules["jax"] = jax_stub


ensure_tensorflow_compat()
import tensorflow as tf

try:
    import mediapipe as mp
except Exception:
    mp = None


CLASSES = [
    "calling",
    "clapping",
    "cycling",
    "dancing",
    "drinking",
    "eating",
    "fighting",
    "hugging",
    "laughing",
    "listening_to_music",
    "running",
    "sitting",
    "sleeping",
    "texting",
    "using_laptop",
]

RULES = {
    "using_laptop": {"warn": "POSTURE ALERT!", "sol": "Straighten your back and look up."},
    "sleeping": {"warn": "WAKE UP!", "sol": "Time to stretch or get some water."},
    "fighting": {"warn": "DANGER DETECTED", "sol": "Violence is not the answer. Relax."},
    "texting": {"warn": "SAFETY WARNING", "sol": "Do not text while moving."},
    "sitting": {"warn": "SEDENTARY ALERT", "sol": "Stand up and move for a minute."},
}

INPUT_SIZE = (128, 128)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATHS = [
    os.path.join(BASE_DIR, "model", "action_model.h5"),
    os.path.join(BASE_DIR, "action_model.h5"),
]


@st.cache_resource(show_spinner=False)
def load_action_model():
    for model_path in MODEL_PATHS:
        if os.path.exists(model_path):
            return tf.keras.models.load_model(model_path), model_path
    return None, None


@st.cache_resource(show_spinner=False)
def load_pose():
    if mp is None:
        return None, None, None

    solutions = getattr(mp, "solutions", None)
    if solutions is None:
        return None, None, None

    pose = solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    return solutions.pose, pose, solutions.drawing_utils


def preprocess(frame_rgb):
    image = cv2.resize(frame_rgb, INPUT_SIZE)
    image = image.astype("float32") / 255.0
    return np.expand_dims(image, axis=0)


def predict_action(model, frame_rgb):
    if model is None:
        return "sitting", 0.95

    prediction = model.predict(preprocess(frame_rgb), verbose=0)
    index = int(np.argmax(prediction))
    confidence = float(prediction[0][index])
    return CLASSES[index], confidence


def draw_pose(frame_rgb):
    mp_pose, pose, mp_draw = load_pose()
    if pose is None:
        return frame_rgb

    output = frame_rgb.copy()
    results = pose.process(output)
    if results.pose_landmarks:
        mp_draw.draw_landmarks(output, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    return output


def image_file_to_rgb(image_file):
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        return None
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def render_prediction(action, confidence):
    alert = RULES.get(action) if confidence > 0.75 else None

    col1, col2 = st.columns(2)
    col1.metric("Detected Activity", action.replace("_", " ").title())
    col2.metric("Confidence", f"{confidence * 100:.1f}%")

    if alert:
        st.error(alert["warn"])
        st.info(f"Fix: {alert['sol']}")
    else:
        st.success("No unsafe activity detected.")


def main():
    st.set_page_config(page_title="Human Activity Recognition", page_icon="AI", layout="wide")

    st.title("Human Activity Recognition")
    st.caption("Upload an image or capture one from your browser camera to detect the activity.")

    with st.spinner("Loading model..."):
        model, model_path = load_action_model()

    if model_path:
        st.sidebar.success(f"Model loaded: {os.path.relpath(model_path, BASE_DIR)}")
    else:
        st.sidebar.warning("Model file not found. App is running in demo mode.")

    st.sidebar.header("Input")
    input_mode = st.sidebar.radio("Choose image source", ["Camera", "Upload Image"])
    show_pose = st.sidebar.checkbox("Show pose landmarks", value=True)

    image_file = None
    if input_mode == "Camera":
        image_file = st.camera_input("Take a picture")
    else:
        image_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

    if image_file is None:
        st.info("Capture or upload an image to start prediction.")
        return

    frame_rgb = image_file_to_rgb(image_file)
    if frame_rgb is None:
        st.error("Could not read this image. Please try another file.")
        return

    action, confidence = predict_action(model, frame_rgb)
    display_frame = draw_pose(frame_rgb) if show_pose else frame_rgb

    image_col, result_col = st.columns([1.2, 1])
    with image_col:
        st.image(display_frame, caption="Input image", use_container_width=True)
    with result_col:
        render_prediction(action, confidence)


if __name__ == "__main__":
    main()
