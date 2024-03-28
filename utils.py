import matplotlib.pyplot as plt
# def get_param(shape):
# 	param = Parameter(torch.Tensor(*shape)); 	
# 	xavier_normal_(param.data)
# 	return param
import mindspore
import numpy
import numpy as np
import pandas as pd
from mindspore import Tensor, ops
from mindspore.common.initializer import XavierNormal, initializer

# import torch
# from torch.nn.init import xavier_normal_
# from torch.nn import functional as F
# from torch.nn import Parameter


def get_param(shape):
	param = mindspore.Parameter(shape=shape, dtype=mindspore.float32)
	return param

def generate_linear(dims, act=None, norm_type=None, norm_dim=None):
    layers = []
    for i in range(len(dims)-1):
        layers.append(mindspore.nn.Dense(dims[i], dims[i+1]))
        # if norm_type is not None:
        #     if norm_type == 'bn':
        #         if norm_dim is None:
        #             layers.append(mindspore.nn.BatchNorm1d(dims[i+1]))
        #         else:
        #             layers.append(mindspore.nn.BatchNorm1d(norm_dim))
        #     elif norm_type == 'ln':
        #         if norm_dim is None:
        #             layers.append(mindspore.nn.LayerNorm([dims[i+1]]))
        #         else:
        #             layers.append(mindspore.nn.LayerNorm([norm_dim]))
        # if act is not None:
        layers.append(act)
    return mindspore.nn.SequentialCell(*layers)

def show_multi_curve(ys, title, legends, xxlabel, yylabel, if_point = False):
    x = np.array(range(len(ys[0])))
    for i in range(len(ys)):
        if if_point:
            plt.plot(x, ys[i], label = legends[i], marker = 'o')
        else:
            plt.plot(x, ys[i], label = legends[i])   
    plt.axis()
    plt.title(title)
    plt.xlabel(xxlabel)
    plt.ylabel(yylabel)
    plt.legend()
    plt.show()
    
def show_curve(ys, xxlabel, yylabel, title):
    x = np.array(range(len(ys)))
    y = np.array(ys)
    plt.plot(x, y, c='b')
    plt.axis()
    plt.title('{} curve'.format(title))
    plt.xlabel(xxlabel)
    plt.ylabel(yylabel)
    plt.show()