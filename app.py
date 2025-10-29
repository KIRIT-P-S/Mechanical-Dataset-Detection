import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

import torch.nn as nn
import timm


class CNNFeatureExtractor(nn.Module):
    def __init__(self, output_dim=768):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))
        self.feature_projection = nn.Linear(512, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        features = self.conv_blocks(x)
        features = self.adaptive_pool(features)
        # Flatten spatial dim
        features = features.view(features.size(0), 512, -1).transpose(1, 2)  # B, 196, 512
        features = self.feature_projection(features)
        features = self.dropout(features)
        return features

class CNNDeiTModel(nn.Module):
    def __init__(self, num_classes=15, embed_dim=768):
        super().__init__()
        self.cnn_extractor = CNNFeatureExtractor(output_dim=embed_dim)
        self.deit_model = timm.create_model('deit_base_patch16_224', pretrained=True, num_classes=0)
        self.transformer_blocks = self.deit_model.blocks
        self.norm = self.deit_model.norm
        self.pos_embed = nn.Parameter(torch.zeros(1, 197, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim // 2, num_classes)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.dist_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        cnn_features = self.cnn_extractor(x)  # B, 196, 768
        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_tokens = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_tokens, cnn_features), dim=1)  # B, 198, 768
        pos_embed = torch.cat(
            (self.pos_embed[:, :1], self.pos_embed[:, :1], self.pos_embed[:, 1:]), dim=1
        )
        x = x + pos_embed[:, : x.size(1)]
        for block in self.transformer_blocks:
            x = block(x)
        x = self.norm(x)
        cls_output = x[:, 0]
        output = self.classifier(cls_output)
        return output



categories = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metalnut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"
]

@st.cache_resource
def load_model():
    device = "cpu"
    model = CNNDeiTModel(num_classes=len(categories))
    model.load_state_dict(torch.load('best_cnn_deit_model.pth', map_location=device))
    model.eval()
    return model

model = load_model()

def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = img.convert('RGB')
    return transform(img).unsqueeze(0)

st.title("CNN–DeiT Hybrid Diet Model Inference")
st.write("Upload an image to classify it using your trained CNN–DeiT hybrid model.")

file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if file is not None:
    image = Image.open(file)
    st.image(image, caption='Uploaded Image', width=128)
    st.write("Classifying...")
    img_tensor = preprocess_image(image)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1).numpy().flatten()
        pred_idx = int(np.argmax(probs))
        st.write(f"### Predicted Class: {categories[pred_idx]}")
        st.write(f"**Confidence:** {probs[pred_idx]:.2f}")