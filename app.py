import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import io

# --- Streamlit page config ---
st.set_page_config(page_title="Neural Style Transfer", page_icon="🎨", layout="centered")

st.markdown(
    "<h1 style='text-align: center; color: #6C63FF;'>🎨 Neural Style Transfer App</h1>", 
    unsafe_allow_html=True
)
st.markdown("<p style='text-align: center;'>Transform any photo into a painting in the style of your favorite artwork!</p>", unsafe_allow_html=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image preprocessing
def image_transform(image):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0).to(device)

def tensor_to_pil(tensor):
    image = tensor.cpu().clone().squeeze(0)
    return transforms.ToPILImage()(image)

def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(c, h * w)
    return torch.mm(features, features.t()) / (c * h * w)

def get_features(x, model, layers):
    features = {}
    i = 0
    for layer in model.children():
        x = layer(x)
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
            if name in layers:
                features[name] = x
    return features

def style_transfer(content_img, style_img, num_steps=200, style_weight=1e6, content_weight=1):
    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    content_features = get_features(content_img, vgg, content_layers)
    content_features = {k: v.detach() for k, v in content_features.items()}
    style_features = get_features(style_img, vgg, style_layers)
    style_grams = {layer: gram_matrix(style_features[layer].detach()) for layer in style_layers}

    target = content_img.clone().requires_grad_(True)
    optimizer = optim.Adam([target], lr=0.003)

    for i in range(num_steps):
        optimizer.zero_grad()
        target_features = get_features(target, vgg, style_layers + content_layers)

        content_loss = torch.mean((target_features['conv_4'] - content_features['conv_4']) ** 2)
        style_loss = 0
        for layer in style_layers:
            target_gram = gram_matrix(target_features[layer])
            style_gram = style_grams[layer]
            style_loss += torch.mean((target_gram - style_gram) ** 2)

        total_loss = content_weight * content_loss + style_weight * style_loss
        total_loss.backward()
        optimizer.step()

    return tensor_to_pil(target)

# --- SIDEBAR ---
st.sidebar.title("🧾 Upload your images")

content_file = st.sidebar.file_uploader("Upload Content Image", type=["jpg", "png", "jpeg"])
style_file = st.sidebar.file_uploader("Upload Style Image", type=["jpg", "png", "jpeg"])

st.sidebar.markdown("---")
style_weight = st.sidebar.slider("🎭 Style Weight", 1e5, 1e7, 1e6, step=1e5)
content_weight = st.sidebar.slider("🧍 Content Weight", 1.0, 10.0, 1.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.info("Developed as part of the Prodigy InfoTech Internship.")

# --- MAIN CONTENT ---
if content_file and style_file:
    content_image = Image.open(content_file).convert('RGB')
    style_image = Image.open(style_file).convert('RGB')

    st.image([content_image, style_image], caption=["Content Image", "Style Image"], width=300)

    if st.button("✨ Generate Stylized Image"):
        with st.spinner("Working hard to paint your masterpiece... 🧠🎨"):
            content_tensor = image_transform(content_image)
            style_tensor = image_transform(style_image)
            output_image = style_transfer(content_tensor, style_tensor, style_weight=style_weight, content_weight=content_weight)

        st.success("✅ Done! Here's your styled image:")
        st.image(output_image, caption="Stylized Output", use_column_width=True)

        buf = io.BytesIO()
        output_image.save(buf, format="JPEG")
        byte_im = buf.getvalue()

        st.download_button(
            label="📥 Download Image",
            data=byte_im,
            file_name="stylized_output.jpg",
            mime="image/jpeg"
        )

else:
    st.warning("👈 Please upload both a content image and a style image from the sidebar.")
