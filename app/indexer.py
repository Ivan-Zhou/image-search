from constants import DATA_DIR, INDEX_LOOKUP_FILE, INDEX_DIR, IMAGE_DIR
import glob
import os
from PIL import Image
import torch
import clip
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def get_index_path(full_image_path):
    image_base_filename = os.path.basename(full_image_path)
    index_path = os.path.join(
        INDEX_DIR, os.path.splitext(image_base_filename)[0]) + ".pt"
    return index_path

# Pre-compute features for all images inside IMAGE_DIR. 
# Results are stored in INDEX_DIR as torch.Tensor.
# Image file path -> Index file path mapping is stored in INDEX_LOOKUP_FILE.

def main():
    IMAGE_DIR = os.path.join(DATA_DIR, 'images')
    if not os.path.exists(IMAGE_DIR):
        raise Exception("Cannot find IMAGE_DIR")

    if not os.path.exists(INDEX_DIR):
        os.mkdir(INDEX_DIR)

    index_data = []
    for image_path in glob.glob(f'{IMAGE_DIR}/*.JPEG'):
        image_input = preprocess(Image.open(
            image_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            # torch.Size([1, 512])
            image_features = model.encode_image(image_input).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)

        index_path = get_index_path(image_path)
        torch.save(image_features, index_path)
        index_data.append((image_path, index_path))
    df = pd.DataFrame(index_data, columns=['image_path', 'index_path'])
    df.to_csv(INDEX_LOOKUP_FILE, index=False)


if __name__ == "__main__":
    main()
