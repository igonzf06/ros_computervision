### Installation
```shell
$ python3 -m <env_name> /path/to/new/virtual/environment
$ virtualenv <env_name>
$ source <env_name>/bin/activate
$ (<env_name>)$ pip install -r path/to/requirements.txt
```

### Pre-trained CNN comparison

| Architectures |  Dataset  |  Accuracy(%) |  Loss  |
| :---          |   :---:   |  :---:       | :---:  |
| VGG-16        |  Imagenet | 81.55        | 3.92   |
| VGG-16        |   Places  | 92.99        | 2.91   |
| ResNet-50     |  Imagenet |   85.92      | 2.11   |
| Inception-V3  |  Imagenet |   61.16      | 6.72   |
| MobileNet     |  Imagenet |   78.64      | 1.41   |

Results from [jupyter notebook](Compare_models.ipynb)


### Usage

You can use the best model by downloading it from [Room classification pre-trained model](https://www.kaggle.com/datasets/irenegonzlezfernndez/pretrained-room-classification)