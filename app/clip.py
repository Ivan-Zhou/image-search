import numpy as np

from transformers import CLIPProcessor, CLIPModel
from visualization import read_image


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
