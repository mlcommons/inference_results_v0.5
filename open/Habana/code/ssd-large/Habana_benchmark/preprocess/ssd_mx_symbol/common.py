'''

Parts were adapted from https://github.com/zhreshold/mxnet-ssd/blob/master/symbol/common.py
This is mathematicly equivelent mxnet implementation - weights are imported from MLPerf resnet34-ssd1200.onnx model

'''
import mxnet as mx
import numpy as np
import  math
import sys
import importlib

from mxnet.symbol import FullyConnected
from mxnet.symbol import Pooling
from mxnet.symbol import Convolution
from mxnet.symbol import Activation
from mxnet.symbol import broadcast_mul
from mxnet.symbol import L2Normalization
from mxnet.symbol import concat as Concat
from mxnet.symbol import softmax

from mxnet.symbol import Flatten
name_generator = mx.name.NameManager()



def conv_act_layer(from_layer, name, num_filter, kernel=(1,1), pad=(0,0), \
    stride=(1,1), act_type="relu",no_bias=False):
    """
    wrapper for a small Convolution group

    Parameters:
    ----------
    from_layer : mx.symbol
        continue on which layer
    name : str
        base name of the new layers
    num_filter : int
        how many filters to use in Convolution layer
    kernel : tuple (int, int)
        kernel size (h, w)
    pad : tuple (int, int)
        padding size (h, w)
    stride : tuple (int, int)
        stride size (h, w)
    act_type : str
        activation type, can be relu...
    use_batchnorm : bool
        whether to use batch normalization

    Returns:
    ----------
    (conv, relu) mx.Symbols
    """
    bias = mx.symbol.Variable(name="{}_bias".format(name),
        init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0'})
    conv = Convolution(data=from_layer, kernel=kernel, pad=pad, \
        stride=stride, num_filter=num_filter, name=name, bias=None if no_bias else bias,no_bias=no_bias)
    act = mx.symbol.Activation(data=conv, act_type=act_type, name="{}_{}".format(name, act_type))
    return act

def conv_act_layer_old(from_layer, name, num_filter, kernel=(1,1), pad=(0,0), \
    stride=(1,1), act_type="relu", use_batchnorm=False,tf_pad=False,in_shape=None,no_bias=False):
    """
    wrapper for a small Convolution group

    Parameters:
    ----------
    from_layer : mx.symbol
        continue on which layer
    name : str
        base name of the new layers
    num_filter : int
        how many filters to use in Convolution layer
    kernel : tuple (int, int)
        kernel size (h, w)
    pad : tuple (int, int)
        padding size (h, w)
    stride : tuple (int, int)
        stride size (h, w)
    act_type : str
        activation type, can be relu...
    use_batchnorm : bool
        whether to use batch normalization

    Returns:
    ----------
    (conv, relu) mx.Symbols
    """
    bias = mx.symbol.Variable(name="{}_conv_bias".format(name),
        init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0'})
    if tf_pad:
        assert in_shape, 'must provide input shape to simulate tensorflow SAME padding'
        from_layer,out_shape,pad = same_pad(from_layer,in_shape,kernel,stride)
    

    conv = Convolution(data=from_layer, kernel=kernel, pad=pad, \
        stride=stride, num_filter=num_filter, name="{}_conv".format(name), bias=None if no_bias else bias,no_bias=no_bias)
    if use_batchnorm:
        conv = mx.symbol.BatchNorm(data=conv,fix_gamma=False, name="{}_bn".format(name))
    if act_type == 'relu6':
        act = mx.symbol.clip(conv,0,6,"{}_relu6".format(name))
    else:
        act = mx.symbol.Activation(data=conv, act_type=act_type, \
            name="{}_{}".format(name, act_type))
    if in_shape:
        return act,out_shape
    else:
        return act


def multi_layer_feature(body, from_layers, num_filters, strides, pads, min_filter=128,tf_pad=False,in_shapes=None,
                        act_type='relu',absorb_bn=False,use_batchnorm=False,multi_feat_no_bias=False,reshape_like_tf=False,**kwargs):
    """Wrapper function to extract features from base network, attaching extra
    layers and SSD specific layers

    Parameters
    ----------
    from_layers : list of str
        feature extraction layers, use '' for add extra layers
        For example:
        from_layers = ['relu4_3', 'fc7', '', '', '', '']
        which means extract feature from relu4_3 and fc7, adding 4 extra layers
        on top of fc7
    num_filters : list of int
        number of filters for extra layers, you can use -1 for extracted features,
        however, if normalization and scale is applied, the number of filter for
        that layer must be provided.
        For example:
        num_filters = [512, -1, 512, 256, 256, 256]
    strides : list of int
        strides for the 3x3 convolution appended, -1 can be used for extracted
        feature layers
    pads : list of int
        paddings for the 3x3 convolution, -1 can be used for extracted layers
    min_filter : int
        minimum number of filters used in 1x1 convolution

    Returns
    -------
    list of mx.Symbols

    """
    # arguments check
    assert len(from_layers) > 0
    assert isinstance(from_layers[0], str) and len(from_layers[0].strip()) > 0
    assert len(from_layers) == len(num_filters) == len(strides) == len(pads)

    internals = body.get_internals()
    layers = []
    no_bias = False if absorb_bn else multi_feat_no_bias
    use_batchnorm = not absorb_bn and use_batchnorm
    for k, params in enumerate(zip(from_layers, num_filters, strides, pads)):
        from_layer, num_filter, s, p = params
        if from_layer.strip():
            # extract from base network
            layer = internals[from_layer.strip() + '_output']
            layers.append(layer)
        else:
            # attach from last feature layer
            assert len(layers) > 0
            assert num_filter > 0
            layer = layers[-1]
            num_1x1 = max(min_filter, num_filter // 2)
            
            name='backbone.additional_blocks.%d.%d'%(k-1,0)
            conv_1x1 = conv_act_layer(layer, name,
                num_1x1, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type=act_type,
                                      no_bias=no_bias)

            name='backbone.additional_blocks.%d.%d'%(k-1,2)
            conv_3x3 = conv_act_layer(conv_1x1, name,
                num_filter, kernel=(3, 3), pad=(p, p), stride=(s, s), act_type=act_type,
                 no_bias=no_bias)
            layers.append(conv_3x3)
    return layers

def multibox_layer(from_layers, num_classes, sizes=[.2, .95],
                   ratios=[1], normalization=-1, num_channels=[],
                   clip=False, interm_layer=0, steps=[],
                   transpose_cat=True, ext_anchors=None, anchors_per_scale=None,prob_type='softmax',
                   detector_kernel=(3,3),detector_padding=(1,1),detector_stride=(1,1),
                   no_bias=False,**kwargs):
    """
    the basic aggregation module for SSD detection. Takes in multiple layers,
    generate multiple object detection targets by customized layers

    Parameters:
    ----------
    from_layers : list of mx.symbol
        generate multibox detection from layers
    num_classes : int
        number of classes excluding background, will automatically handle
        background in this function
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    num_channels : list of int
        number of input layer channels, used when normalization is enabled, the
        length of list should equals to number of normalization layers
    clip : bool
        whether to clip out-of-image boxes
    interm_layer : int
        if > 0, will add a intermediate Convolution layer
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions

    Returns:
    ----------
    list of outputs, as [loc_preds, cls_preds, anchor_boxes]
    loc_preds : localization regression prediction
    cls_preds : classification prediction
    anchor_boxes : generated anchor boxes
    """
    assert len(from_layers) > 0, "from_layers must not be empty list"
    assert num_classes > 0, \
        "num_classes {} must be larger than 0".format(num_classes)

    assert len(ratios) > 0, "aspect ratios must not be empty list"
    if not isinstance(ratios[0], list):
        # provided only one ratio list, broadcast to all from_layers
        ratios = [ratios] * len(from_layers)
    assert len(ratios) == len(from_layers), \
        "ratios and from_layers must have same length"

    assert len(sizes) > 0, "sizes must not be empty list"
    if len(sizes) == 2 and not isinstance(sizes[0], list):
        # provided size range, we need to compute the sizes for each layer
        assert sizes[0] > 0 and sizes[0] < 1
        assert sizes[1] > 0 and sizes[1] < 1 and sizes[1] > sizes[0]
        tmp = np.linspace(sizes[0], sizes[1], num=(len(from_layers)-1))
        min_sizes = [start_offset] + tmp.tolist()
        max_sizes = tmp.tolist() + [tmp[-1]+start_offset]
        sizes = zip(min_sizes, max_sizes)
    assert len(sizes) == len(from_layers), \
        "sizes and from_layers must have same length"

    if not isinstance(normalization, list):
        normalization = [normalization] * len(from_layers)
    assert len(normalization) == len(from_layers)

    assert sum(x > 0 for x in normalization) <= len(num_channels), \
        "must provide number of channels for each normalized layer"

    if steps:
        assert len(steps) == len(from_layers), "provide steps for all layers or leave empty"

    loc_pred_layers = []
    cls_pred_layers = []
    anchor_layers = []
    num_classes += 1 # always use background as label 0

    for k, from_layer in enumerate(from_layers):
        from_name = from_layer.name
        # normalize
        if normalization[k] > 0:
            from_layer = L2Normalization(data=from_layer, \
                                         mode="channel", name="{}_norm".format(from_name))
            scale = mx.symbol.Variable(name="{}_scale".format(from_name),
                                       shape=(1, num_channels.pop(0), 1, 1),
                                       init=mx.init.Constant(normalization[k]),
                                       attr={'__wd_mult__': '0.1'})
            from_layer = mx.symbol.broadcast_mul(lhs=scale, rhs=from_layer)
        if interm_layer > 0:
            from_layer = Convolution(data=from_layer, kernel=(3,3), \
                                     stride=(1,1), pad=(1,1), num_filter=interm_layer, \
                                     name="{}_inter_conv".format(from_name))
            from_layer = Activation(data=from_layer, act_type="relu", \
                                    name="{}_inter_relu".format(from_name))

        # estimate number of anchors per location
        # here I follow the original version in caffe
        # TODO: better way to shape the anchors??
        size = sizes[k]
        assert len(size) > 0, "must provide at least one size"
        size_str = "(" + ",".join([str(x) for x in size]) + ")"
        ratio = ratios[k]
        assert len(ratio) > 0, "must provide at least one ratio"
        ratio_str = "(" + ",".join([str(x) for x in ratio]) + ")"
        if not anchors_per_scale:
            num_anchors = len(size) -1 + len(ratio)
        else:
            num_anchors = anchors_per_scale[k]


        # create location prediction layer
        num_loc_pred = num_anchors * 4
        name = 'backbone.loc.%d'%(k)
        bias = mx.symbol.Variable(name=name+'_bias',
                                  init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0'})
        loc_pred = Convolution(data=from_layer, bias=None if no_bias else bias,no_bias=no_bias, kernel=detector_kernel,\
                               stride=detector_stride, pad=detector_padding, num_filter=num_loc_pred, \
                               name=name)
        if transpose_cat:
            loc_pred = mx.symbol.transpose(loc_pred, axes=(0,2,3,1))
            loc_pred = Flatten(data=loc_pred,name='flatten_loc_preds_{}'.format(k))
        else:
            loc_pred = loc_pred.reshape((0, 4, -1),name='reshape_{}'.format(loc_pred.name))

        loc_pred_layers.append(loc_pred)

        # create class prediction layer
        num_cls_pred = num_anchors * num_classes
        name='backbone.conf.%d'%(k)
        
        bias = mx.symbol.Variable(name=name+'_bias',
                                  init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0'})
        cls_pred = Convolution(data=from_layer, bias=None if no_bias else bias,no_bias=no_bias, kernel=detector_kernel, \
                               stride=detector_stride, pad=detector_padding, num_filter=num_cls_pred, \
                               name=name)
        if transpose_cat:
            # usual mxnet-ssd case, channels are in the fast changing dim
            cls_pred = mx.symbol.transpose(cls_pred, axes=(0, 2, 3, 1))
            cls_pred = mx.symbol.Reshape(data=cls_pred, shape=(0, -1, num_classes),
                                         name='reshape_{}'.format(cls_pred.name))
        else:
            # mlperf onnx model replaces the nhwc transpose with simple reshape,
            # class predictions should be [B,#class,#anchors], but apx softmax expect
            # the classes in the last dimension, thus we always transpose for softmax on last dim then transpose again
            cls_pred = mx.symbol.Reshape(data=cls_pred, shape=(0, num_classes, -1),
                                         name='reshape_{}'.format(cls_pred.name))
        assert prob_type in ['softmax','sigmoid'], 'prob type can only be in [softmax,sigmoid] got {}'.format(prob_type)

        
        # float case
        
        if transpose_cat:
            if prob_type == 'softmax':
                cls_prob = softmax(data=cls_pred, name='{}_cls_prob'.format(from_name), axis=-1)
            elif prob_type == 'sigmoid':
                cls_prob = Activation(data=cls_pred, act_type='sigmoid', name='{}_cls_prob'.format(from_name))

            cls_prob = mx.symbol.transpose(cls_prob, axes=(0, 2, 1), name='{}_transpose_out'.format(from_name))
        else:
            if prob_type == 'softmax':
                name
                cls_prob = softmax(data=cls_pred, name='{}_cls_prob'.format(from_name), axis=1)
            elif prob_type == 'sigmoid':
                cls_prob = Activation(data=cls_pred, act_type='sigmoid', name='{}_cls_prob'.format(from_name))

        # prob concat now on dim 2
        cls_pred_layers.append(cls_prob)
        if ext_anchors is None:
            # create anchor generation layer
            if steps:
                step = (steps[k], steps[k])
            else:
                step = '(-1.0, -1.0)'

            anchors = mx.contrib.symbol.MultiBoxPrior(from_layer, sizes=size_str, ratios=ratio_str, \
                                                      clip=clip, name="{}_anchors".format(from_name), steps=step)
            anchors = Flatten(data=anchors)
            anchor_layers.append(anchors)

    if ext_anchors is None:
        anchor_boxes = Concat(*anchor_layers, dim=1)
        anchor_boxes = mx.symbol.Reshape(data=anchor_boxes, shape=(0, -1, 4), name="multibox_anchors")
    else:
        # overwrite with external anchors
        anchor_boxes = mx.symbol.Variable('multibox_anchors', shape=ext_anchors.shape,
                                          init=mx.init.Constant(ext_anchors.tolist()))

    # this is how the float model will look without the additional nodes for i16 softmax
    loc_preds = Concat(*loc_pred_layers, dim=1 if transpose_cat else 2, name="multibox_loc_pred")
    cls_preds = Concat(*cls_pred_layers, dim=2, name='cls_pred_concat')

    return [loc_preds, cls_preds, anchor_boxes]



