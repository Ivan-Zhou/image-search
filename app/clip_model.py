import numpy as np
import torch
import clip as openai_clip
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from visualization import read_image
import streamlit as st
import faiss


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
            inputs = self.processor(
                text=[query], images=image, return_tensors="pt", padding=True
            )
            outputs = self.model(**inputs)
            # this is the image-text similarity score
            logits_per_image = outputs.logits_per_image
            score = logits_per_image.cpu().detach().numpy()[0][0]
            scores[i] = score

        df = pd.DataFrame()
        df["score"] = scores
        df["image_path"] = image_paths
        return df


# OpenAI CLIP model
class CLIPOpenAI:
    def __init__(self, index_lookup_file=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = openai_clip.load("ViT-B/32", device=self.device)
        self.index_lookup_file = index_lookup_file

    def get_similarity_scores(self, image_paths, query):
        if self.index_lookup_file:
            # Use image indexing. Loading image_features from saved torch tensors.
            df = pd.read_csv(self.index_lookup_file)
            image_features = []
            for image_path in image_paths:
                selected_row = df.loc[df["image_path"] == image_path].iloc[0]
                saved_path = selected_row["index_path"]
                img_feature = torch.load(saved_path)
                image_features.append(img_feature)
            image_features = torch.stack(image_features).squeeze(1).to(self.device)

        else:
            # Calculate image features from scratch.
            # Followed colab on https://github.com/openai/CLIP
            images = []
            for image_path in image_paths:
                image = self.preprocess(Image.open(image_path).convert("RGB"))
                images.append(image)

            image_input = torch.tensor(np.stack(images)).to(self.device)
            with torch.no_grad():
                # torch.Size([N, 512])
                image_features = self.model.encode_image(image_input).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)

        text_tokens = openai_clip.tokenize([query]).to(self.device)
        with torch.no_grad():
            # torch.Size([1, 512])
            text_features = self.model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
        # Shape (N, )
        scores = np.squeeze(similarity) * 100

        df = pd.DataFrame()
        df["score"] = scores
        df["image_path"] = image_paths
        return df


# OpenAI CLIP model
class CLIPOpenAIFaiss:
    def build_index(self):
        if self.image_paths is None:
            return

        st.info("Re-building FAISS index for new data...", icon="ℹ️")

        # IndexFlatIP: Exact Search for Inner Product
        self.index = faiss.IndexFlatIP(512)

        # Use image indexing. Loading image_features from saved torch tensors.
        df = pd.read_csv(self.index_lookup_file)

        image_features = []
        for image_path in self.image_paths:
            selected_row = df.loc[df["image_path"] == image_path].iloc[0]
            saved_path = selected_row["index_path"]
            img_feature = torch.load(saved_path)
            image_features.append(img_feature)
        # Shape: (N, 512)
        image_features = (
            torch.stack(image_features).squeeze(1).to(self.device).cpu().numpy()
        )
        self.index.add(image_features)

    def __init__(self, index_lookup_file, image_paths, k_neighbors=5):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = openai_clip.load("ViT-B/32", device=self.device)
        self.index_lookup_file = index_lookup_file
        self.k_neighbors = k_neighbors
        self.image_paths = image_paths

        self.build_index()

    def get_similarity_scores(self, image_paths, query):
        if not np.array_equal(image_paths, self.image_paths):
            self.image_paths = image_paths
            self.build_index()

        text_tokens = openai_clip.tokenize([query]).to(self.device)
        with torch.no_grad():
            # torch.Size([1, 512])
            text_features = self.model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        # Shape (1, 512)
        text_features = text_features.cpu().numpy()

        # Find nearest k_neighbors. similarity, indexes has Shape (1, k_neighbors)
        similarity, indexes = self.index.search(text_features, self.k_neighbors)

        df = pd.DataFrame()
        df["image_path"] = np.take(self.image_paths, indexes[0], axis=0)
        df["score"] = similarity[0] * 100
        return df
