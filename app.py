import os
import gdown

MODEL_PATH = "Emotion_Recognition_ConvNeXtV2.pth"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(
        id="1SGUV-FbJfB9XxKsHfgGWdUHXo2XRshcC",
        output=MODEL_PATH,
        quiet=False
    )
import streamlit as st
import torch
import torch.nn as nn
import timm
import numpy as np
import pandas as pd

from PIL import Image
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

# ----------------------------
# Configuration
# ----------------------------

st.set_page_config(
    page_title="Emotion Detection",
    page_icon="😊",
    layout="centered"
)

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


# ----------------------------
# Load Model
# ----------------------------

@st.cache_resource
def load_model():

    model = timm.create_model(
        "convnextv2_base.fcmae_ft_in22k_in1k",
        pretrained=False,
        num_classes=7
    )

    # Recreate custom classifier exactly as during training
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
        "Emotion_Recognition_ConvNeXtV2.pth",
        map_location=device
    )

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model


model = load_model()


# ----------------------------
# Image Transform
# ----------------------------

transform = Compose([
    Resize(224,224),
    Normalize(
        mean=(0.485,0.456,0.406),
        std=(0.229,0.224,0.225)
    ),
    ToTensorV2()
])


# ----------------------------
# Streamlit UI
# ----------------------------

st.title("😊 Emotion Detection System")

st.write(
    "Upload a face image and the trained ConvNeXtV2 model will predict the emotion."
)

uploaded = st.file_uploader(
    "Upload Image",
    type=["jpg","jpeg","png","webp"]
)

if uploaded is not None:

    image = Image.open(uploaded).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    image_np = np.array(image)

    tensor = transform(image=image_np)["image"]
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(tensor)

        probs = torch.softmax(output, dim=1)

        confidence, pred = torch.max(probs,1)

    st.success(
        f"Predicted Emotion : **{emotion_names[pred.item()]}**"
    )

    st.write(
        f"Confidence : **{confidence.item()*100:.2f}%**"
    )

    st.subheader("Prediction Probabilities")

    df = pd.DataFrame({
        "Emotion":[emotion_names[i] for i in range(7)],
        "Probability":probs.cpu().numpy()[0]
    })

    st.bar_chart(
        df.set_index("Emotion")
    )

    st.subheader("Top 3 Predictions")

    values, indices = torch.topk(probs,3)

    for score, idx in zip(values[0], indices[0]):

        st.write(
            f"**{emotion_names[idx.item()]}** : {score.item()*100:.2f}%"
        )
