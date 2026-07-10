import os
import gdown
import streamlit as st
import torch
import torch.nn as nn
import timm
import numpy as np
import pandas as pd

from PIL import Image
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

# -----------------------------
# Streamlit Page
# -----------------------------

st.set_page_config(
    page_title="Emotion Detection System",
    page_icon="😊",
    layout="centered"
)

# -----------------------------
# Google Drive Model
# -----------------------------

FILE_ID = "1SGUV-FbJfB9XxKsHfgGWdUHXo2XRshcC"

MODEL_PATH = "Emotion_Recognition_ConvNeXtV2.pth"

# -----------------------------
# Download model only once
# -----------------------------

if not os.path.exists(MODEL_PATH):

    with st.spinner("Downloading trained model (first launch only)..."):

        gdown.download(
            id=FILE_ID,
            output=MODEL_PATH,
            quiet=False
        )

# -----------------------------
# Class names
# -----------------------------

emotion_names = {
    0: "Surprise",
    1: "Fear",
    2: "Disgust",
    3: "Happy",
    4: "Sad",
    5: "Angry",
    6: "Neutral"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load Model
# -----------------------------

@st.cache_resource
def load_model():

    model = timm.create_model(
        "convnextv2_base.fcmae_ft_in22k_in1k",
        pretrained=False,
        num_classes=7
    )

    model.head.fc = nn.Sequential(

        nn.Linear(1024,1024),
        nn.GELU(),
        nn.BatchNorm1d(1024),
        nn.Dropout(0.4),

        nn.Linear(1024,512),
        nn.GELU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),

        nn.Linear(512,256),
        nn.GELU(),
        nn.Dropout(0.2),

        nn.Linear(256,7)

    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(state_dict)

    model.to(device)

    model.eval()

    return model

model = load_model()

# -----------------------------
# Image Transform
# -----------------------------

transform = Compose([

    Resize(224,224),

    Normalize(

        mean=(0.485,0.456,0.406),

        std=(0.229,0.224,0.225)

    ),

    ToTensorV2()

])

# -----------------------------
# UI
# -----------------------------

st.title("😊 Emotion Detection System")

st.write(
    "Upload a face image and the ConvNeXtV2 model will predict the emotion."
)

uploaded = st.file_uploader(
    "Choose an image",
    type=["jpg","jpeg","png","webp"]
)

if uploaded is not None:

    image = Image.open(uploaded).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    img = np.array(image)

    tensor = transform(image=img)["image"]

    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(tensor)

        probs = torch.softmax(output,1)

    confidence, prediction = torch.max(probs,1)

    st.success(
        f"Predicted Emotion: {emotion_names[prediction.item()]}"
    )

    st.write(
        f"Confidence: {confidence.item()*100:.2f}%"
    )

    values, indices = torch.topk(probs,3)

    st.subheader("Top 3 Predictions")

    top3 = pd.DataFrame({

        "Emotion":[emotion_names[i.item()] for i in indices[0]],

        "Confidence (%)":[round(v.item()*100,2) for v in values[0]]

    })

    st.table(top3)

    st.subheader("Probability Distribution")

    chart = pd.DataFrame({

        "Probability":probs.cpu().numpy()[0]

    },

    index=[emotion_names[i] for i in range(7)])

    st.bar_chart(chart)
