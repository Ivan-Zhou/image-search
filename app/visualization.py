import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image


def read_image(path):
    return Image.open(path)


def plot_image(path, title=None):
    assert os.path.exists(path), f"File {path} does not exist"
    img = read_image(path)
    plt.figure(figsize=(5, 5))
    if title:
        plt.title(title)
    plt.axis("off")
    plt.imshow(img)


def plot_multiple_images(images, titles, cols=4, per_image_size=(4, 4)):
    n_images = len(images)
    rows = (n_images + cols - 1) // cols
    fig = plt.figure(
        figsize=(cols * per_image_size[0], rows * per_image_size[1])
    )  # width, height
    for idx, (image, title) in enumerate(zip(images, titles)):
        fig.add_subplot(rows, cols, idx + 1)
        plt.imshow(image)
        plt.title(title)
        plt.axis("off")


def plot_bboxes(image, bboxes, scores, labels):
    """
    Given an image and bounding boxes, plots the image with the bounding boxes
    """
    for bbox, score, label in zip(bboxes, scores, labels):
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"{score:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (36, 255, 12),
            2,
        )
    return image
