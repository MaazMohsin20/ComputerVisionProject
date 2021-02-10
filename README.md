# ComputerVisionProject
# CBAM-Keras
This is a Keras implementation of ["CBAM: Convolutional Block Attention Module"](https://arxiv.org/pdf/1807.06521).
## Prerequisites
- Python 3.x
- Keras
## Prepare Data set
This repository use [*Radiography-Database*](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database) dataset.
## Train a Model
You can simply train a model with `main.py`.

1. Set a model you want to train.
    - e.g. `model = resnet_v1.resnet_v1(input_shape=input_shape, depth=depth, attention_module=attention_module)`  
2. Set attention_module parameter
    - e.g. `attention_module = 'cbam_block'`
3. Set other parameter such as *batch_size*, *epochs*, *data_augmentation* and so on.
4. Run the `main.py` file
    - e.g. `python main.py`
    
## Test Model
Model can be tested by compiling model and load weights by assigning valid weight file path to wightpath in TestWeightsModel.py

