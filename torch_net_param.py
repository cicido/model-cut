# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import torchfile
import pickle

def load_net_param(param_file):
    net_param = None
    with open(param_file,'rb') as f:
        print('load...')
        net_param = pickle.load(f)
    return net_param

def freeze_net_param(net_param,param_file):
    with open(param_file,'wb') as f:
        print('freezing net_param...')
        pickle.dump(net_param,f,protocol = 4)
        
def extract_net_param_from_t7(t7_file, param_file):
    '''
    Loads a Torch network from a saved .t7 file into Tensorflow.
    :param t7 Path to t7 file to use
    :param param_file param to construct tensorflow graph
    '''
    t7 = torchfile.load(t7_file, force_8bytes_long=True)
    net_param = {}
    for idx, module in enumerate(t7.modules):
        if idx > 30:
            break
        if module._typename == b'nn.SpatialReflectionPadding':
            left = module.pad_l
            right = module.pad_r
            top = module.pad_t
            bottom = module.pad_b
            type_dict = net_param.setdefault(idx, {})
            type_dict['pad'] = [[0, 0], [top, bottom], [left, right], [0, 0]]

        elif module._typename == b'nn.SpatialConvolution':
            weight = module.weight.transpose([2, 3, 1, 0])
            bias = module.bias
            strides = [1, module.dH, module.dW, 1]  # Assumes 'NHWC'

            type_dict = net_param.setdefault(idx, {})
            type_dict['conv'] = [weight, strides, bias]

        elif module._typename == b'nn.ReLU':
            type_dict = net_param.setdefault(idx, {})
            type_dict['relu'] = []

        elif module._typename == b'nn.SpatialUpSamplingNearest':
            type_dict = net_param.setdefault(idx, {})
            type_dict['susn'] = module.scale_factor
            
        elif module._typename == b'nn.SpatialMaxPooling':
            type_dict = net_param.setdefault(idx, {})
            type_dict['pool'] = [[1, module.kH, module.kW, 1], [1, module.dH, module.dW, 1], str(module.name, 'utf-8')]

        else:
            raise NotImplementedError(module._typename)
     
    freeze_net_param(net_param,param_file)
    return net_param

def construct_net(net,graph,param_file):
    net_param = load_net_param(param_file)
    conv_map = {}
    
    with graph.as_default():
        for i in sorted(net_param.keys()):
            type_dict = net_param[i]
            for item in type_dict.items():
                key = item[0]
                if 'pad' == key:
                    net = tf.pad(net,item[1], 'REFLECT')
                elif 'conv' == key:
                    weight = item[1][0]
                    strides = item[1][1]
                    bias = item[1][2]
                    net = tf.nn.conv2d(net, weight, strides, padding='VALID')
                    conv_map[net.name.split(':')[0]] = i
                    net = tf.nn.bias_add(net, bias)
                elif 'relu' == key:
                    net = tf.nn.relu(net)
                elif 'susn' == key:
                    d = tf.shape(net)
                    scale_factor = item[1]
                    #print(scale_factor)
                    size = [d[1] * scale_factor, d[2] * scale_factor]
                    net = tf.image.resize_nearest_neighbor(net, size)
                elif 'pool' == key:
                    ksize = item[1][0]
                    strides = item[1][1]
                    name = item[1][2]
                    net = tf.nn.max_pool(net, ksize=ksize, strides=strides,padding='VALID', name = name)
    return net,conv_map
