import numpy as np
import mxnet as mx
import itertools
from math import sqrt, ceil
from ssd_mx_symbol.common import multi_layer_feature, multibox_layer

class DefaultBoxes(object):
    def __init__(self, fig_size, feat_size, steps, scales, aspect_ratios, \
                       scale_xy=0.1, scale_wh=0.2):

        self.feat_size = feat_size
        self.fig_size_w,self.fig_size_h = fig_size

        self.scale_xy_ = scale_xy
        self.scale_wh_ = scale_wh
        
        # According to https://github.com/weiliu89/caffe
        # Calculation method slightly different from paper
        self.steps_w = [st[0] for st in steps]
        self.steps_h = [st[1] for st in steps]
        self.scales = scales
        fkw = self.fig_size_w//np.array(self.steps_w)
        fkh = self.fig_size_h//np.array(self.steps_h)
        self.aspect_ratios = aspect_ratios

        self.default_boxes = []
        # size of feature and number of feature
        for idx, sfeat in enumerate(self.feat_size):
            sfeat_w,sfeat_h=sfeat
            sk1 = scales[idx][0]/self.fig_size_w
            sk2 = scales[idx+1][1]/self.fig_size_h
            sk3 = sqrt(sk1*sk2)
            all_sizes = [(sk1, sk1), (sk3, sk3)]
            for alpha in aspect_ratios[idx]:
                w, h = sk1*sqrt(alpha), sk1/sqrt(alpha)
                all_sizes.append((w, h))
                all_sizes.append((h, w))
            for w, h in all_sizes:
                for i, j in itertools.product(range(sfeat_w), range(sfeat_h)):
                    cx, cy = (j+0.5)/fkh[idx], (i+0.5)/fkw[idx]
                    self.default_boxes.append((cx, cy, w, h)) 
        self.dboxes = np.array(self.default_boxes)
        self.dboxes = self.dboxes.clip(min=0, max=1)
        # For IoU calculation
        self.dboxes_ltrb = np.copy(self.dboxes)
        self.dboxes_ltrb[:, 0] = self.dboxes[:, 0] - 0.5*self.dboxes[:, 2]
        self.dboxes_ltrb[:, 1] = self.dboxes[:, 1] - 0.5*self.dboxes[:, 3]
        self.dboxes_ltrb[:, 2] = self.dboxes[:, 0] + 0.5*self.dboxes[:, 2]
        self.dboxes_ltrb[:, 3] = self.dboxes[:, 1] + 0.5*self.dboxes[:, 3]
    
    @property
    def scale_xy(self):
        return self.scale_xy_
    
    @property    
    def scale_wh(self):
        return self.scale_wh_

    def __call__(self, order="ltrb"):
        if order == "ltrb": return self.dboxes_ltrb
        if order == "xywh": return self.dboxes

def get_scales(min_scale=0.2, max_scale=0.9,num_layers=6):
    """ Following the ssd arxiv paper, regarding the calculation of scales & ratios

    Parameters
    ----------
    min_scale : float
    max_scales: float
    num_layers: int
        number of layers that will have a detection head
    anchor_ratios: list
    first_layer_ratios: list

    return
    ------
    sizes : list
        list of scale sizes per feature layer
    ratios : list
        list of anchor_ratios per feature layer
    """

    # this code follows the original implementation of wei liu
    # for more, look at ssd/score_ssd_pascal.py:310 in the original caffe implementation
    min_ratio = int(min_scale * 100)
    max_ratio = int(max_scale * 100)
    step = int(np.floor((max_ratio - min_ratio) / (num_layers - 2)))
    min_sizes = []
    max_sizes = []
    for ratio in range(min_ratio, max_ratio + 1, step):
        min_sizes.append(ratio / 100.)
        max_sizes.append((ratio + step) / 100.)
    min_sizes = [int(100*min_scale / 2.0) / 100.0] + min_sizes
    max_sizes = [min_scale] + max_sizes

    # convert it back to this implementation's notation:
    scales = []
    for layer_idx in range(num_layers):
        scales.append([min_sizes[layer_idx], np.single(np.sqrt(min_sizes[layer_idx] * max_sizes[layer_idx]))])
    return scales

def pytorch_dboxes_1200(figsize = (1200,1200),feat_size = [(38,38), (19,19), (10,10), (5,5), (3,3), (1,1)]):
    scales_base = [21, 45, 99, 153, 207, 261, 315]
    steps=[(int(figsize[0]/fs[0]),int(figsize[1]/fs[1])) for fs in feat_size]
    scales = [(int(s*figsize[0]/300),int(s*figsize[1]/300)) for s in scales_base]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    return dboxes 


def settings():
    network='resnetv1'
    num_classes=80
    num_layers = 34
    data_shape = 1200
    absorb_bn=False
    image_shape = '3,224,224'  # resnet require it as shape check
    reset_body_strides = {'backbone.model.layer2.0.0.conv1': '(1,1)', 'backbone.model.layer2.0.0.downsample.0': '(1,1)'}
    from_layers = ['backbone.model.layer2.0.5.relu2', '', '', '', '', '']
    num_filters = [-1, 512, 512, 256, 256, 256]
    strides = [-1, 2, 2, 2, 1, 1]
    pads = [-1, 1, 1, 1, 0, 0]
    sizes = get_scales(min_scale=0.2, max_scale=0.9, num_layers=len(from_layers))
    ratios = [[1, 2, .5], [1, 2, .5, 3, 1. / 3], [1, 2, .5, 3, 1. / 3], [1, 2, .5, 3, 1. / 3], \
              [1, 2, .5], [1, 2, .5]]
    normalizations = -1
    bottle_neck = False
    steps = []
    num_anchors = len(sizes) - 1 + len(ratios)  ##per ssd block
    num_anchors = [4, 6, 6, 6, 4, 4]
    fig_size = (data_shape, data_shape)
    strides = [-1, 2, 2, 2, 2, 1]
    detector_stride = (3, 3)
    detector_pad = (1, 1)
    feat_size = [[50, 50], [25, 25], [13, 13], [7, 7], [3, 3], [3, 3]]
    ext_anchors = pytorch_dboxes_1200((data_shape, data_shape), feat_size)

    transpose_cat = False
    ext_anchors = ext_anchors(order='ltrb')
    #ext_anchors = ext_anchors.numpy()
    if ext_anchors.ndim == 2:
        ext_anchors = np.expand_dims(ext_anchors, 0)
    return locals()

def import_module(module_name,path='ssd_mx_symbol'):
    """Helper function to import module"""
    import sys, os
    import importlib
    if path:
        sys.path.append(os.path.join(os.path.dirname(__file__),'ssd_mx_symbol'))
    else:
        sys.path.append(os.path.dirname(__file__))
    return importlib.import_module(module_name)


def get_symbol(network, num_classes, from_layers, num_filters, sizes, ratios,
               strides, pads, normalizations=-1, steps=[], min_filter=128, reset_body_strides=[],
               transpose_cat=True,**kwargs):
    """Build network for testing SSD

    Parameters
    ----------
    network : str
        base network symbol name
    num_classes : int
        number of object classes not including background
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
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions
    min_filter : int
        minimum number of filters used in 1x1 convolution

    Returns
    -------
    mx.Symbol

    """

    body = import_module(network).get_symbol(num_classes=num_classes, **kwargs)
    if len(reset_body_strides)>0:
        # a hack to rewrite stride attribute of a model for some given layer
        import json
        json_string = body.tojson()
        json_net = json.loads(json_string)
        fixup = [ (l,reset_body_strides[l['name']]) for l in json_net['nodes'] if l['name'] in reset_body_strides.keys()]
        for x,s in fixup:
            x['attrs']['stride']=s
        json_string = json.dumps(json_net)
        body = mx.symbol.load_json(json_string)

    layers = multi_layer_feature(body, from_layers, num_filters, strides, pads, min_filter=min_filter,**kwargs)

    loc_preds, cls_prob, anchor_boxes = multibox_layer(layers, num_classes, sizes=sizes, ratios=ratios,
                                                        normalization=normalizations, num_channels=num_filters,
                                                        clip=False, interm_layer=0, steps=steps,
                                                        transpose_cat=transpose_cat,**kwargs)


    # model uses external_post process, outputs should be [B,#class,#anchors],[B,4,#anchors]
    return mx.symbol.Group([cls_prob,loc_preds])

def get_arg_aux_from_param_file(fname):
    p=mx.nd.load(fname)
    a, u = {}, {}
    for k, v in p.items():
        if k.startswith('aux'):
            u[k[4:]] = v
        else:
            a[k[4:]] = v
    return a,u

def bind_mxnet_model(mx_model_path='r34-v1-ssd_1200'):
    s=get_symbol(**settings())
    mod = mx.mod.Module(s, label_names=None, context=mx.cpu())
    mod.bind([mx.io.DataDesc('data', (1, 3, 1200, 1200))])
    mod.init_params()
    mod.save_checkpoint(mx_model_path, 0)
    return mx_model_path
