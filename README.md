# unet-segmentation

Semantic segmentation using **UNet model** on **Cityscape Segmentation dataset**.

## Installation
**Working environments**:
```
Python 3.6.9
Cuda 11.0
Ubuntu 18.10
```
With those settings, I am using `pytorch v1.7.1` to use the GPU.

**Install**

Inside of the `root` directory, execute :
```bash
# Minimal setup
pip install .

# or Dev setup
# pip install .[dev]

# Install pytorch afterwards
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
```

## Data
An example of dataset format is inside the docstring of module **`segmentation/data.py`**

## Usage
You can look at example of usage inside the folder **`scripts/`**.
Otherwise, display the helper description of the main function to get all arguments :
```bash
# from scripts/
python main.py --help
```