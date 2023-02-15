import matplotlib.pyplot as plt
import os
from PIL import Image


def read_image(path):
    return Image.open(path)


def plot_image(path, title=None):
    assert os.path.exists(path), f"File {path} does not exist"
    img = read_image(path)
    plt.figure(figsize = (5, 5))
    if title:
        plt.title(title)
    plt.axis("off")
    plt.imshow(img)


def plot_multiple_images(images, titles, cols=4, per_image_size=(4, 4)):
    n_images = len(images)
    rows = (n_images + cols - 1) // cols
    fig = plt.figure(figsize=(cols * per_image_size[0], rows * per_image_size[1])) # width, height
    for idx, (image, title) in enumerate(zip(images, titles)):
        fig.add_subplot(rows, cols, idx+1)
        plt.imshow(image)
        plt.title(title)
        plt.axis("off")
