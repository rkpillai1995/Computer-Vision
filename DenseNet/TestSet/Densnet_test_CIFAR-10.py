__author__ = 'Rajkumar Pillai'


import torch
import time
import torch.nn as nn

import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
import numpy as np
import matplotlib.pyplot as plt
import numpy
from torchvision import datasets, models, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
import matplotlib.image as pngimg

torch.cuda.empty_cache()

"""
file: Densenet_test_CIFAR-10.py
CSCI-631:  COMPUTER VISION
Author: Rajkumar Lenin Pillai


Description:This program uses the  denset model trained on CIFAR-10 dataset
and predicts the labels of test set and plots the training loss curve
"""

### Transformomg the data in order to use densenet model
data_transform = transforms.Compose(
     [
        transforms.Resize((64, 64)),  ### Resizing image dimenesion to 224x224
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.squeeze(x)),  ### To remove dimeensions of input of size 1
     ])


##Downloading the CIFAR-10 test_set
testset = torchvision.datasets.CIFAR10(root='./data-cifar-10', train=False, download=True, transform=data_transform)

# Loading the test set
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')




import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict


if __name__ == '__main__':

    # Defining the network

    class _DenseLayer(nn.Sequential):
        def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
            super(_DenseLayer, self).__init__()
            self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
            self.add_module('relu1', nn.ReLU(inplace=True)),
            self.add_module('conv1', nn.Conv2d(num_input_features,
                            growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
            self.drop_rate = drop_rate

        def forward(self, x):
            new_features = super(_DenseLayer, self).forward(x)
            if self.drop_rate > 0:
                new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
            return torch.cat([x, new_features], 1)


    class _DenseBlock(nn.Sequential):
        def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
            super(_DenseBlock, self).__init__()
            for i in range(num_layers):
                layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
                self.add_module('denselayer%d' % (i + 1), layer)


    class _Transition(nn.Sequential):
        def __init__(self, num_input_features, num_output_features):
            super(_Transition, self).__init__()
            self.add_module('norm', nn.BatchNorm2d(num_input_features))
            self.add_module('relu', nn.ReLU(inplace=True))
            self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                              kernel_size=1, stride=1, bias=False))
            self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


    class DenseNet(nn.Module):
        r"""Densenet-BC model class, based on
        `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

        Args:
            growth_rate (int) - how many filters to add each layer (`k` in paper)
            block_config (list of 4 ints) - how many layers in each pooling block
            num_init_features (int) - the number of filters to learn in the first convolution layer
            bn_size (int) - multiplicative factor for number of bottle neck layers
              (i.e. bn_size * k features in the bottleneck layer)
            drop_rate (float) - dropout rate after each dense layer
            num_classes (int) - number of classification classes
        """
        def __init__(self, growth_rate=32, block_config=(4,4,4,4),
                     num_init_features=64, bn_size=4, drop_rate=0, num_classes=10):

            super(DenseNet, self).__init__()

            # First convolution
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                ('norm0', nn.BatchNorm2d(num_init_features)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ]))

            # Each denseblock
            num_features = num_init_features
            for i, num_layers in enumerate(block_config):
                block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                    bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
                self.features.add_module('denseblock%d' % (i + 1), block)
                num_features = num_features + num_layers * growth_rate
                if i != len(block_config) - 1:
                    trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                    self.features.add_module('transition%d' % (i + 1), trans)
                    num_features = num_features // 2

            # Final batch norm
            self.features.add_module('norm5', nn.BatchNorm2d(num_features))

            # Linear layer
            self.classifier = nn.Linear(248, num_classes)



        def forward(self, x):
            features = self.features(x)
            out = F.relu(features, inplace=True)
            out = F.avg_pool2d(out, kernel_size=2, stride=2).view(features.size(0),-1)
            out = self.classifier(out)
            return out


    # Using CUDA if we have a GPU that supports it along with the correct
    # Install, otherwise use the CPU
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


    ####################################

    net = DenseNet()
    #print(net)
    ##Loading the trained densenet model
    net.load_state_dict(torch.load('densenet_model_weights_CIFAR-10'))
    net = net.to(device)

    #####################################


    ## Plotting the accuracies
    print("Accuracy calculation")

    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('\nAccuracy of the network on the  test images: %d %%\n' % (
        100 * correct / total))


    ## To view the training loss curve
    img = pngimg.imread('model_loss_CIFAR-10.png')
    plt.imshow(img)
    plt.show()

    ## To view parameters of network
    print(net.parameters)

