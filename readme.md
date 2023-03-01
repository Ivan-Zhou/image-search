# Image Search
## Installation
Create a Conda environment and install the required packages:
```
conda create -n cs324 python=3.10
pip install -r requirements.txt
```

Check out [PyTorch page](https://pytorch.org/get-started/locally/) for the installation guide for your system.
For example, I installed torch-1.13.1+cu116 torchaudio-0.13.1+cu116 torchvision-0.14.1+cu116 for my CUDA environment.
```
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

Install faiss from conda-forge ([details](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md#installing-from-conda-forge)) for faster vector distance search.

**Note: Feel free to skip this installation. Still exploring. Not in the current implementation.**

```
# CPU version
$ conda install -c conda-forge faiss-cpu

# GPU version
$ conda install -c conda-forge faiss-gpu
>>>>>>> main
```

### Install GLIP
Pull the submodule:
```
git submodule update --init --recursive

## Run a local app
The local app is a simple web app that allows you to select a dataset and type a query to search:

![Local app](resources/app_screenshot.png)

Before you run the app, make sure you've prepared the dataset and set up the right path with `DATA_DIR` variable in `app/constants.py`.

Run indexer first for faster image feature retrieval:
```
python app/indexer.py
```

You can run the local app by executing the following command:
```
sh execute.sh
```


## Performance Findings


| Model                     | Search Time (sec) | Image Sample Size | Device            |
| ------------------------- | ----------------- | ----------------- | ----------------- |
| HuggingFace CLIP          | 6.49              | 100               | MacBook Pro - CPU |
| OpenAI CLIP               | 3.35              | 100               | MacBook Pro - CPU |
| OpenAI CLIP - FasterImage | 0.4               | 100               | MacBook Pro - CPU |
