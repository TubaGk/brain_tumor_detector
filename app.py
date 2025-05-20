import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from src.model import CNNModel  # src/model.py içindeki model sınıfı

# Sınıf isimleri
classes = ['no_tumor', 'pituitary', 'glioma', 'meningioma']

# Modeli yükle
def load_model():
    model = CNNModel()
    model.load_state_dict(torch.load("model_weights.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

# Grad-CAM fonksiyonu
def generate_gradcam(model, img_tensor):
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    forward_handle = model.conv2.register_forward_hook(forward_hook)
    backward_handle = model.conv2.register_backward_hook(backward_hook)

    output = model(img_tensor)
    pred_class = output.argmax().item()
    model.zero_grad()
    output[0, pred_class].backward()

    grad = gradients[0].squeeze().detach().cpu().numpy()
    act = activations[0].squeeze().detach().cpu().numpy()

    weights = grad.mean(axis=(1, 2))
    cam = np.zeros(act.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (img_tensor.shape[2], img_tensor.shape[3]))
    cam = (cam - cam.min()) / (cam.max() - cam.min())

    forward_handle.remove()
    backward_handle.remove()

    return cam

# Streamlit arayüzü
def main():
    st.set_page_config(page_title="Beyin Tümörü Tespiti", layout="centered")
    st.title("🧠 Beyin MRI Tümör Tespiti")
    st.write("Bir MRI görüntüsü yükleyin. Model tümör tipini tahmin edecek ve Grad-CAM ile görselleştirecek.")

    uploaded_file = st.file_uploader("🖼️ MRI Görüntüsü Seçin", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Yüklenen MRI", use_column_width=True)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        input_tensor = transform(image).unsqueeze(0)

        model = load_model()

        with st.spinner("🧠 Tahmin yapılıyor..."):
            output = model(input_tensor)
            pred = torch.softmax(output, dim=1)
            pred_class = torch.argmax(pred).item()
            prob = pred[0, pred_class].item()

            cam = generate_gradcam(model, input_tensor)
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            original = np.array(image.resize((224, 224)))
            overlay = cv2.addWeighted(original, 0.5, heatmap, 0.5, 0)

            label = classes[pred_class]
            st.markdown(f"### 🔍 Tahmin: **{label.upper()}**")
            st.markdown(f"### 📊 Güven Skoru: **%{prob * 100:.2f}**")
            st.image(overlay, caption="Grad-CAM ile Görselleştirme", use_column_width=True)

if __name__ == "__main__":
    main()
