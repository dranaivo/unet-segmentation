# unet-segmentation

Semantic segmentation using **UNet model** on **Cityscape Segmentation dataset**.

## Installation
**Working environments**:
```
Python 3.6.9
```

Inside of the `root` directory, execute :
```bash
pip install -r requirements.txt
```

## Data
An example of dataset format is inside the docstring of module **`segmentation/data.py`**

## Usage
You can look at example of usage inside the folder **`scripts/`**.
Otherwise, display the helper description of the main function to get all arguments :
```bash
# from segmentation/
python main.py --help
```

## About testing
I was not able to add testing. Nevertheless, I show a planning of what I consider testing inside each module of the folder **`tests/`**.