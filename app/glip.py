import os

# TODO: better way to import GLIP
os.chdir("./GLIP")

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

from pathlib import Path
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import torch
from time import time

pylab.rcParams["figure.figsize"] = 20, 12
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo

# TODO: better way to import GLIP
os.chdir("../")

GLIP_DIR = "./GLIP"
WEIGHT_PATH = os.path.join(GLIP_DIR, "MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth")
CONFIG_FILE = os.path.join(GLIP_DIR, "configs/pretrain/glip_Swin_T_O365_GoldG.yaml")
REMOTE_PATH = "https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/glip_tiny_model_o365_goldg_cc_sbu.pth"
if not os.path.exists(WEIGHT_PATH):
    Path(WEIGHT_PATH).parent.mkdir(parents=True, exist_ok=True)
    print("Downloading weights...")
    os.system(f"wget {REMOTE_PATH} -O {WEIGHT_PATH}")
    print("Done!")


class GLIP:
    def __init__(self) -> None:
        self._init_cfg(cfg)
        self._check_gpu()
        self._init_glip_model()

    def _init_cfg(self, cfg):
        self.cfg = cfg
        self.cfg.local_rank = 0
        self.cfg.num_gpus = 1
        self.cfg.merge_from_file(CONFIG_FILE)
        self.cfg.merge_from_list(["MODEL.WEIGHT", WEIGHT_PATH])
        self.cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

    def _check_gpu(self):
        try:
            print("Check GPU info:")
            print("CUDA available: {}".format(torch.cuda.is_available()))
            print("CUDA version: {}".format(torch.version.cuda))
            print("GPU count: {}".format(torch.cuda.device_count()))
            print("GPU name: {}".format(torch.cuda.get_device_name(0)))
            print(
                "GPU memory: {}".format(
                    torch.cuda.get_device_properties(0).total_memory
                )
            )
        except Exception as e:
            raise ValueError("Fail to check GPU info due to {}".format(e))

    def _init_glip_model(self):
        print("Initializing GLIP Model...")
        start = time()
        self.glip_model = GLIPDemo(
            self.cfg,
            min_image_size=800,
            confidence_threshold=0.7,
            show_mask_heatmaps=False,
        )
        print("Time: {}".format(time() - start))

    def predict(self, image, query, debug=False):
        predictions = self.glip_model.inference(image, query)
        bboxes = predictions.bbox.tolist()
        scores = predictions.get_field("scores").tolist()
        labels = predictions.get_field("labels").tolist()
        if debug:
            print("bboxes: {}".format(bboxes))
            print("scores: {}".format(scores))
            print("labels: {}".format(labels))
        class_names = query.split(" ")
        if debug:
            print(f"class_names: {class_names}")
        classes = [class_names[label - 1] for label in labels]
        if debug:
            print(f"classes: {classes}")
        return bboxes, scores, classes


def load(url):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    response = requests.get(url)
    pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image


if __name__ == "__main__":
    glip = GLIP()
    image = load("http://farm4.staticflickr.com/3693/9472793441_b7822c00de_z.jpg")
    bboxes, scores, classes = glip.predict(
        image,
        "person with two sofa and a remote controller besides the shelf",
        debug=True,
    )
