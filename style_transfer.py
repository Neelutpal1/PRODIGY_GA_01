import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load image and preprocess
def load_image(path, max_size=512):
    image = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)
    return image.to(device)

# Display tensor as image
def imshow(tensor, title=None):
    image = tensor.cpu().clone().squeeze(0)
    image = transforms.ToPILImage()(image)
    if title:
        plt.title(title)
    plt.imshow(image)
    plt.axis('off')

# Compute Gram matrix for style
def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(c, h * w)
    return torch.mm(features, features.t()) / (c * h * w)

# Extract intermediate features
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

def main():
    # Load images
    content_img = load_image("images/content.jpg")
    style_img = load_image("images/style.jpg")

    # Load VGG19 model
    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()

    # Define layers to use
    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    content_features = get_features(content_img, vgg, content_layers)
    content_features = {k: v.detach() for k, v in content_features.items()}

    style_features = get_features(style_img, vgg, style_layers)
    style_grams = {layer: gram_matrix(style_features[layer].detach()) for layer in style_layers}


    # Initialize target
    target = content_img.clone().requires_grad_(True)
    optimizer = optim.Adam([target], lr=0.003)

    # Loss weights
    style_weight = 1e6
    content_weight = 1e0

    # Optimization loop
    for i in range(1, 501):
        optimizer.zero_grad()

        target_features = get_features(target, vgg, style_layers + content_layers)

        # Content loss
        content_loss = torch.mean((target_features['conv_4'] - content_features['conv_4']) ** 2)

        # Style loss
        style_loss = 0
        for layer in style_layers:
            target_gram = gram_matrix(target_features[layer])
            style_gram = style_grams[layer]
            style_loss += torch.mean((target_gram - style_gram) ** 2)

        # Total loss
        total_loss = content_weight * content_loss + style_weight * style_loss

        total_loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print(f"Iteration {i}, Total Loss: {total_loss.item():.4f}")

    # Save final result
    output = target.cpu().clone().squeeze(0)
    output_img = transforms.ToPILImage()(output)
    os.makedirs("output", exist_ok=True)
    output_img.save("output/stylized_output.jpg")
    print("Saved stylized image to output/stylized_output.jpg")

    # Show final image
    plt.figure()
    imshow(target, title="Stylized Image")
    plt.show()

if __name__ == "__main__":
    main()
