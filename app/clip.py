import numpy as np
import torch
import clip
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from visualization import read_image


# Hugging face CLIP model
class CLIP:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def get_similarity_scores(self, image_paths, query):
        # https://huggingface.co/docs/transformers/model_doc/clip#usage
        scores = np.zeros(len(image_paths))
        for i, image_path in enumerate(image_paths):
            image = read_image(image_path)
            inputs = self.processor(text=[query], images=image, return_tensors="pt", padding=True)
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
            score = logits_per_image.cpu().detach().numpy()[0][0]
            scores[i] = score
        return scores

# OpenAI CLIP model
class CLIPOpenAI():
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
    
    def get_similarity_scores(self, image_paths, query):
        # Followed colab on https://github.com/openai/CLIP
        images = []
        for image_path in image_paths:
            image = self.preprocess(Image.open(image_path).convert("RGB"))
            images.append(image)

        image_input = torch.tensor(np.stack(images)).to(self.device)
        text_tokens = clip.tokenize([query]).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_input).float()
            text_features = self.model.encode_text(text_tokens).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
        # Shape (N, ) 
        return np.squeeze(similarity) * 100