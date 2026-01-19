#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import tempfile
import os
import seaborn as sns

# --- CONFIGURATION ---
IMG_SIZE = (128, 128)
MODEL_PATH = "plant_disease_model.h5"

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# --- EXTRACT CLASS NAMES ---
class_names = list(model.predict(tf.zeros((1,) + IMG_SIZE + (3,)), verbose=0)[0].shape)
st.title("🌿 Plant Disease Classification with Grad-CAM")

# --- HELPER: get base submodel (e.g., MobileNetV2) ---
def get_base_model(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer
    raise ValueError("No nested model found.")

# --- GRAD-CAM FUNCTION ---
def make_gradcam_heatmap(img_array, model, base_model, pred_index=None):
    conv_layers = [l for l in base_model.layers if isinstance(l, tf.keras.layers.Conv2D)]
    last_conv_layer = conv_layers[-1]

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index.numpy()) if isinstance(pred_index, tf.Tensor) else int(pred_index)

# --- DISPLAY FUNCTION ---
def show_gradcam(img_path, model):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr, verbose=0)[0]
    pred_idx = np.argmax(preds)
    confidence = preds[pred_idx]

    base_model = get_base_model(model)
    heatmap, _ = make_gradcam_heatmap(arr, model, base_model, pred_idx)

    # Grad-CAM overlay
    img_orig = cv2.imread(img_path)
    img_orig = cv2.resize(img_orig, IMG_SIZE)
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_resized = cv2.resize(heatmap_uint8, (img_orig.shape[1], img_orig.shape[0]))
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_orig, 0.6, heatmap_color, 0.4, 0)

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax1.set_title(f"Grad-CAM Overlay\nConfidence: {confidence:.2f}")
    ax1.axis("off")

    top_k = 5
    top_indices = preds.argsort()[-top_k:][::-1]
    top_labels = [str(i) for i in top_indices]
    top_scores = preds[top_indices]
    sns.barplot(x=top_scores[::-1], y=top_labels[::-1], palette="viridis", ax=ax2)
    ax2.set_title("Top-5 Predicted Classes")
    ax2.set_xlabel("Confidence")
    plt.tight_layout()
    st.pyplot(fig)

    return pred_idx, confidence

# --- STREAMLIT UI ---
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # save temporarily
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(uploaded_file.read())
    st.image(uploaded_file, caption="Uploaded image", use_column_width=True)
    st.write("Processing...")

    try:
        pred_idx, conf = show_gradcam(temp.name, model)
        st.success(f"Predicted class index: {pred_idx} (Confidence: {conf:.2f})")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
    finally:
        os.remove(temp.name)


# In[ ]:




