# unet-segmentation

Semantic segmentation using **UNet model** on **Cityscape Segmentation dataset**.

## Installation
**Working environments**:
```
Python 3.6.9
```

First, install **PyTorch**:
```bash
pip install torch==1.9.0+cpu torchvision==0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```
Then :
```bash
pip install -r requirements.txt
```

## Data
An example of dataset format is inside the docstring of module **`segmentation/data.py`**

## Usage
You can look at example of usage inside the folder **`scripts`**.
Otherwise, display the helper description of the main function to get all arguments :
```bash
# from segmentation/
python main.py --help
```

## About testing