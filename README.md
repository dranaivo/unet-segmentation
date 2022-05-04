# unet-segmentation

Semantic segmentation using **UNet model** on **Cityscape Segmentation dataset**.

## Installation
**Working environments**:
```
python 3.6.9
pip 21.3.1
setuptools 59.6.0
cuda 11.0
ubuntu 18.10
```
>With those settings, I am using `pytorch v1.7.1` to use the GPU.

**Install**

First clone the repository. Then, inside of the `root` directory, execute :
```bash
# Minimal setup
pip install .

# or Dev setup
# pip install .[dev]
```

Finally, install `pytorch v1.7.1` and `torchvision v0.8.2` using **`pip`** (instructions on the [official website](https://pytorch.org/get-started/previous-versions/)).

## Data
An example of dataset format is inside the docstring of module **`segmentation/data.py`**

## Usage
You can look at example of usage inside the folder **`scripts/`**.
Otherwise, display the helper description of the main function to get all arguments :
```bash
# from scripts/
python main.py --help
```