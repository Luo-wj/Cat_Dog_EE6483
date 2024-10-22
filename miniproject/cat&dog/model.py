import torch
import importlib

import sys
sys.path.append("/models")

net=importlib.import_module('models.CNN')
CNN=net.CNN
net=importlib.import_module('models.ResNet')
ResNet=net.ResNet
net=importlib.import_module('models.ResNet_CBAM')
ResNet_CBAM=net.ResNet_CBAM


def model_generator(method):
    if method == 'CNN':
        model = CNN()
    elif method == 'ResNet':
        model = ResNet()
    elif method == 'ResNet_CBAM':
        model = ResNet_CBAM()
    else:
        print(f'Method {method} is not defined !!!!')

    return model
