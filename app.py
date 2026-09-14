import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from attacks.fgsm import fgsm_attack


# PAGE CONFIG

st.set_page_config(page_title="Adversarial ML Pro", layout="wide")

st.title("🔐 Adversarial Attack Dashboard")
st.markdown("### Breaking & Securing Deep Learning Models")


# SIDEBAR

st.sidebar.title("⚙️ Settings")

epsilon = st.sidebar.slider("Attack Strength (Epsilon)", 0.0, 0.3, 0.1)

model_type = st.sidebar.radio(
    "Choose Model",
    ["Defense Model", "Baseline Model"]
)

st.sidebar.markdown("""
**Models:**
- Baseline → Vulnerable  
- Defense → Robust  
""")


# LOAD MODEL

@st.cache_resource
def load_model(path):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

if model_type == "Defense Model":
    model = load_model("models/defense_model.pth")
else:
    model = load_model("models/baseline_model.pth")


# TRANSFORM

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

classes = ["Airplane", "Automobile", "Bird", "Cat", "Deer",
           "Dog", "Frog", "Horse", "Ship", "Truck"]

# IMAGE INPUT

st.subheader("📸 Input Image")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])
camera_image = st.camera_input("Or Capture from Webcam")

image = None

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

elif camera_image:
    image = Image.open(camera_image).convert("RGB")


# PROCESS IMAGE

if image is not None:

    img = transform(image).unsqueeze(0)
    img.requires_grad = True

    # ORIGINAL
    output = model(img)
    pred = output.argmax().item()
    confidence = torch.softmax(output, dim=1)[0][pred].item()

    # LOSS
    loss = nn.CrossEntropyLoss()(output, torch.tensor([pred]))
    model.zero_grad()
    loss.backward()

    data_grad = img.grad.data

    # ATTACK
    adv_img = fgsm_attack(img, epsilon, data_grad)

    adv_output = model(adv_img)
    adv_pred = adv_output.argmax().item()
    adv_conf = torch.softmax(adv_output, dim=1)[0][adv_pred].item()

    
    # DISPLAY IMAGES
    
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original", use_container_width=True)

    with col2:
        adv_np = adv_img.squeeze().detach().permute(1,2,0).numpy()
        st.image(adv_np, caption="Adversarial", use_container_width=True)

    
    # PREDICTIONS
   
    st.markdown(f"### 🟢 Original: {classes[pred]} ({confidence:.2f})")
    st.markdown(f"### 🔴 Adversarial: {classes[adv_pred]} ({adv_conf:.2f})")

    
    # RESULT
    
    if pred != adv_pred:
        st.error("⚠️ Model Fooled!")
    else:
        st.success("✅ Model Robust")

    # GRAPH
    
    st.subheader("📊 Confidence vs Epsilon")

    eps_vals = np.linspace(0, 0.3, 6)
    confidences = []

    for eps in eps_vals:
        adv_temp = fgsm_attack(img, eps, data_grad)
        out = model(adv_temp)
        conf = torch.softmax(out, dim=1)[0][out.argmax()].item()
        confidences.append(conf)

    fig, ax = plt.subplots()
    ax.plot(eps_vals, confidences, marker='o')
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Confidence")
    ax.set_title("Model Confidence under Attack")

    st.pyplot(fig)