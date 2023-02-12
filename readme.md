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