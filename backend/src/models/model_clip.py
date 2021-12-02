import torch
import torch.nn as nn
import numpy as np
import clip
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
import requests
from io import BytesIO

result_converter = {1: 'anti-vaxx', 0: 'pro-vaxx'}

class ImageTextClassifier(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.relu = nn.ReLU()
        self.CLIP_model, self.image_preprocess = clip.load("ViT-B/32", device=device)
        self.classifier = nn.Linear(self.CLIP_model.token_embedding.embedding_dim*2, 1)
    
    def tokenize(self, text):
        return clip.tokenize(text, truncate=True).to(self.device)

    def preprocess(self, image):
        return self.image_preprocess(image)

    def forward(self, image, text):
        with torch.no_grad():
            image_features = self.CLIP_model.encode_image(image)
            text_features = self.CLIP_model.encode_text(text)
        image_text = torch.cat([image_features, text_features], 1).float()
        image_text = self.relu(image_text).float()
        logits = self.classifier(image_text)
        return logits

def init_clip():
    model_path = './src/models/model_weights/clip_image_text.pkl'
    model = ImageTextClassifier('cpu').to('cpu')
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def run_clip_inference(tweet_text, img_url, model):
    response = requests.get(img_url)
    pillow_img = Image.open(BytesIO(response.content)).convert('rgb')

    img = model.preprocess(pillow_img)

    # text_data = ["sorry to inform you 'vaccinated' folks, you are the experiment!"]
    text_data = [tweet_text]
    outputs = model(img, text_data)
    preds = torch.max(outputs, 1)[1]

    return result_converter[int(preds[0])]