# -*- coding:utf-8 -*-
import AdaIN
import matplotlib.pyplot as plt
import importlib
importlib.reload(AdaIN)
import construct_input_data
import os
import get_x_w_y
from lib.decompose import dictionary
import torch_net_param
importlib.reload(get_x_w_y)

dataDir = '/data/mydata/val/ilsvrc2012_val'
dataPickle = "input/image_val_data.pickle"
if os.path.isfile(dataPickle):
    print('%s exists' %dataPickle)
else:
    construct_input_data.freeze_image_data(dataDir,dataPickle)
input_data = construct_input_data.load_image_data(dataPickle)


#抽取t7模型参数，以构建网络
vgg_t7_file = 'models/vgg_normalised.t7'
gt_param_file = 'models/param-0.pickle'
if os.path.isfile(gt_param_file):
    print('%s exists' %gt_param_file)
else:
    torch_net_param.extract_net_param_from_t7(vgg_t7_file,gt_param_file)

def modify_net(input_net_param_file,input_data,top_name,bottom_name):
    feats_Y, points_dict,W,conv_map = get_x_w_y.get_gtY(gt_param_file,input_data,top_name)
    feats_X = get_x_w_y.get_X(input_net_param_file,input_data,top_name,points_dict)
    print(conv_map)

    feats_W = W.transpose((3,2,0,1)) #(n,c,h,w)
    feats_X = feats_X.transpose((0,3,1,2)) #(N,c,h,w)
    output = dictionary(feats_X,feats_W,feats_Y,rank = int(feats_W.shape[1]/4))
    idx = output[0]
    newW = output[1].transpose(2,3,1,0)
    newB = output[2]
    net_param = torch_net_param.load_net_param(input_net_param_file)
    top_idx = conv_map[top_name]
    bottom_idx = conv_map[bottom_name]
    print("top_idx,bootom_idx:",top_idx,bottom_idx)

    print("bottom size:")
    bottom_val = net_param[bottom_idx]['conv']
    print(bottom_val[0].shape)
    bottom_val[0] = bottom_val[0][...,idx]
    print(bottom_val[0].shape)
    bottom_val[2] = bottom_val[2][idx]
    print(bottom_val[2].shape)

    print("top size:") 
    top_val = net_param[top_idx]['conv']
    print(top_val[0].shape)
    print(top_val[2].shape)
    top_val[0] = newW
    top_val[2] = top_val[2] + newB
    print(top_val[0].shape)
    print(top_val[2].shape)

    outfile = '.'.join([input_net_param_file,top_name,bottom_name])
    torch_net_param.freeze_net_param(net_param,outfile)
    return outfile

conv_dict = {'Conv2D': 0, 'Conv2D_1': 2, 'Conv2D_2': 5, 'Conv2D_3': 9, 'Conv2D_4': 12, 'Conv2D_5': 16, 'Conv2D_6': 19, 'Conv2D_7': 22, 'Conv2D_8': 25, 'Conv2D_9': 29}

conv_layer = sorted(conv_dict.keys())[1:]
top_bottom_pair_list = zip(conv_layer[1:],conv_layer[:-1])
outfile = 'models/param-0.pickle'
for top_bottom in top_bottom_pair_list:
    top_name,bottom_name = top_bottom
    print(top_name,bottom_name)
    outfile = modify_net(outfile,input_data,top_name,bottom_name)
print(outfile)    
