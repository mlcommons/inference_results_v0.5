'''
Adapted from https://github.com/tornadomeet/ResNet/blob/master/symbol_resnet.py
Original author Wei Wu

Implemented the following paper:

Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Identity Mappings in Deep Residual Networks"
'''
import mxnet as mx

'''
Adapted from https://github.com/tornadomeet/ResNet/blob/master/symbol_resnet.py
Original author Wei Wu

Implemented the following paper:

Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Identity Mappings in Deep Residual Networks"
'''
import mxnet as mx
import numpy as np

from mxnet.symbol import FullyConnected
from mxnet.symbol import Pooling
from mxnet.symbol import Convolution
from mxnet.symbol import Activation
from mxnet.symbol import elemwise_add
from mxnet.symbol import concat as Concat
from mxnet.symbol import softmax
from mxnet.symbol import BatchNorm

from mxnet.symbol import Flatten
from mxnet.symbol import SoftmaxOutput



def residual_unit(data, num_filter, stride, dim_match, name, bn_mom=0.9, workspace=256, bottle_neck =True,memonger=False):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tupe
        Stride used in convolution
    dim_match : Boolen
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    if bottle_neck:
        conv0 = Convolution(data=data, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1,1), pad=(0, 0),
                               no_bias=True, workspace=workspace, name=name + '.conv0')

        bn0 = BatchNorm(data=conv0,name=name + '.bn0',fix_gamma=False,eps=1e-5,momentum=bn_mom)

        act0 = Activation(data=bn0, act_type='relu', name=name + '.relu0')
    else:
        act0 = data

    conv1 = Convolution(data=act0, num_filter=int(num_filter * 0.25) if bottle_neck else num_filter, kernel=(3, 3), stride=stride, pad=(1, 1),
                               no_bias=True, workspace=workspace, name=name + '.conv1')

    bn1= BatchNorm(data=conv1,name=name + '.bn1',fix_gamma=False,eps=1e-5,momentum=bn_mom)

    act1 = Activation(data=bn1, act_type='relu', name=name + '.relu1')
    conv2 = Convolution(data=act1, num_filter=num_filter, kernel=(1, 1) if bottle_neck else (3,3), stride=(1, 1), pad=(0, 0) if bottle_neck else (1,1), no_bias=True,
                               workspace=workspace, name=name + '.conv2')

    bn2 = BatchNorm(data=conv2,name=name + '.bn2',fix_gamma=False,eps=1e-5,momentum=bn_mom)

    if not dim_match:
        shortcut = Convolution(data=data, num_filter=num_filter, kernel=(1, 1) , stride=stride, no_bias=True,
                                           workspace=workspace, name=name + '.downsample.0')
        shortcut=BatchNorm(data=shortcut,name=name + '.downsample.1',fix_gamma=False,eps=1e-5,momentum=bn_mom)
    else:
        shortcut=data
    if memonger:
        shortcut._set_attr(mirror_stage='True')

    res_add = elemwise_add(lhs=shortcut, rhs=bn2)
    return Activation(data=res_add, act_type='relu', name=name + '.relu2')

def resnet(units, num_stages, filter_list, num_classes,image_shape=(3,224,224), bottle_neck=True, bn_mom=0.9, workspace=256, memonger=False,**kwargs):
    """Return ResNet symbol of
    Parameters
    ----------
    units : list
        Number of units in each stage
    num_stages : int
        Number of stage
    filter_list : list
        Channel size of each stage
    num_classes : int
        Ouput size of symbol
    dataset : str
        Dataset type, only cifar10 and imagenet supports
    workspace : int
        Workspace used in convolution operator
    """
    num_unit = len(units)
    assert(num_unit == num_stages)
    data = mx.symbol.Variable(name='data')
    #data = mx.symbol.identity(data=data,name='input')
    (nchannel, height, width) = image_shape

    body = Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2, 2), pad=(3, 3),
                              no_bias=True, name="backbone.model.layer1.0", workspace=workspace)
    
    body = BatchNorm(data=body,name='backbone.model.layer1.1',fix_gamma=False,eps=1e-5,momentum=bn_mom)
    body = Activation(data=body, act_type='relu', name='relu0')
    body = Pooling(data=body, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max')

    for i in range(num_stages):
        a=1 if i<2 else i
        k=i+4 if i<2 else 0        
        name = 'backbone.model.layer%d.%d.%d' % (a,k, 0)
        body = residual_unit(body, filter_list[i + 1], (1,1) if i == 0 else (2,2), filter_list[i]==filter_list[i+1],
                             name=name, bottle_neck=bottle_neck, workspace=workspace,
                             memonger=memonger)
        for j in range(units[i] - 1):
            name = 'backbone.model.layer%d.%d.%d' % (a,k, j + 1) 
            body = residual_unit(body, filter_list[i + 1], (1, 1), True, name=name,
                                 bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)

    pool1 = Pooling(data=body, global_pool=True, kernel=(
        7, 7), pool_type='avg', name='pool1')
    flat = Flatten(data=pool1)

    fc1 = FullyConnected(
        data=flat, num_hidden=num_classes, name='fc')

    return SoftmaxOutput(data=fc1, name='softmax')


def get_symbol(num_classes, num_layers, image_shape, conv_workspace=256,bottle_neck=None, **kwargs):
    """
    Adapted from https://github.com/tornadomeet/ResNet/blob/master/train_resnet.py
    Original author Wei Wu
    """
    image_shape = [int(l) for l in image_shape.split(',')]
    (nchannel, height, width) = image_shape
    if height <= 28:
        num_stages = 3
        if (num_layers - 2) % 9 == 0 and num_layers >= 164:
            per_unit = [(num_layers - 2) // 9]
            filter_list = [16, 64, 128, 256]
            bottle_neck = True if bottle_neck is None else bottle_neck
        elif (num_layers - 2) % 6 == 0 and num_layers < 164:
            per_unit = [(num_layers - 2) // 6]
            filter_list = [16, 16, 32, 64]
            bottle_neck = False if bottle_neck is None else bottle_neck
        else:
            raise ValueError(
                "no experiments done on num_layers {}, you can do it youself".format(num_layers))
        units = per_unit * num_stages
    else:
        if num_layers >= 50:
            filter_list = [64, 256, 512, 1024, 2048]
            bottle_neck = True if bottle_neck is None else bottle_neck
        else:
            filter_list = [64, 64, 128, 256, 512]
            bottle_neck = False if bottle_neck is None else bottle_neck
        num_stages = 4
        if num_layers == 18:
            units = [2, 2, 2, 2]
        elif num_layers == 34:
            units = [3, 4, 6, 3]
        elif num_layers == 50:
            units = [3, 4, 6, 3]
        elif num_layers == 101:
            units = [3, 4, 23, 3]
        elif num_layers == 152:
            units = [3, 8, 36, 3]
        elif num_layers == 200:
            units = [3, 24, 36, 3]
        elif num_layers == 269:
            units = [3, 30, 48, 8]
        else:
            raise ValueError(
                "no experiments done on num_layers {}, you can do it youself".format(num_layers))

    return resnet(units=units,
                  num_stages=num_stages,
                  filter_list=filter_list,
                  num_classes=num_classes,
                  image_shape=image_shape,
                  bottle_neck=bottle_neck,
                  workspace=conv_workspace,**kwargs)