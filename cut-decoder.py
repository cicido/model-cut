import AdaIN
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import importlib
importlib.reload(AdaIN)
import construct_input_data
import os
import get_x_w_y
from lib.decompose import dictionary
import torch_net_param
importlib.reload(get_x_w_y)
'''
print('*'*20 + "load image load" + '*' * 20)
dataDir = '/data/mydata/val/ilsvrc2012_val'
dataPickle = "input/image_val_data.pickle"
if os.path.isfile(dataPickle):
    print('%s exists' %dataPickle)
else:
    construct_input_data.freeze_image_data(dataDir,dataPickle)
input_data = construct_input_data.load_image_data(dataPickle)
print('load end!!')
'''
#进一步计算decoder的输入，即AdaIN的输出.
def get_AdaIN_input(input_data, alpha, vgg_t7_file):
    input_shape = input_data.shape
    nPicsPerBatch = input_shape[1]
    nBatches = input_shape[0]
    nPics = nPicsPerBatch * nBatches
    with tf.Graph().as_default() as g, tf.Session(graph=g) as sess, tf.variable_scope(tf.get_variable_scope(), reuse=False) as scope:
        c_holder = tf.placeholder(tf.float32,[1,None,None,None],name="input_content")
        s_holder = tf.placeholder(tf.float32,[1,None,None,None],name="input_style")
        print(c_holder)
        print(s_holder)
        
        c_vgg,_ = torch_net_param.construct_net(c_holder, g, vgg_t7_file)
        s_vgg,_ = torch_net_param.construct_net(s_holder, g, vgg_t7_file)
 
        stylized_content = AdaIN.AdaIN(c_vgg, s_vgg, alpha)
        print("sc_shape:",stylized_content.shape)
        is_first = True
        output_data = None
        
        for i in range(nPics):
            print("i:",i)
            batch_idx = np.random.randint(nBatches,size=(2,))
            pics_idx = np.random.randint(nPicsPerBatch,size=(2,))
            c = np.expand_dims(input_data[batch_idx[0]][pics_idx[0]],axis=0)
            s = np.expand_dims(input_data[batch_idx[1]][pics_idx[1]],axis=0)
            #print("c.shape:",c.shape)
            #print("s.shape:",s.shape)
            res = sess.run(stylized_content,feed_dict={c_holder:c,s_holder:s})
            print("res.shape:",res.shape)       
            a,b = divmod(i,nPicsPerBatch)
            if is_first:
                output_shape = [nBatches,nPicsPerBatch] + list(res.shape[1:])
                output_data = np.ndarray(shape=output_shape)
                is_first = False
            output_data[a][b] = np.squeeze(res, axis=0)
        return output_data

print('*'*20 + "decoder_input_data" + '*'*20)
decoder_dataPickle = "input/decoder_image_val_data.pickle"
decoder_input_data = None
if os.path.isfile(decoder_dataPickle):
    print('%s exists' %decoder_dataPickle)
    decoder_input_data = construct_input_data.load_image_data(decoder_dataPickle)
else:
    vgg_t7 = 'models/param-0.pickle'
    alpha = 1
    decoder_input_data = get_AdaIN_input(input_data, alpha, vgg_t7)
    construct_input_data.freeze_data(decoder_input_data,decoder_dataPickle)
    
import torch_net_param
importlib.reload(torch_net_param)
print('*'*20 + '抽取decoder.t7' + '*'*20)
#抽取t7模型参数，抽取后的结果保存到文件gt_param_file,以构建网络
decoder_file = 'models/decoder.t7'
gt_param_file = 'models/decoder-param-0.pickle'
if os.path.isfile(gt_param_file):
    print('%s exists' %gt_param_file)
else:
    torch_net_param.extract_net_param_from_t7(decoder_file,gt_param_file)

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

#conv_dict = {'Conv2D': 0, 'Conv2D_1': 2, 'Conv2D_2': 5, 'Conv2D_3': 9, 'Conv2D_4': 12, 'Conv2D_5': 16, 'Conv2D_6': 19, 'Conv2D_7': 22, 'Conv2D_8': 25, 'Conv2D_9': 29}
conv_dict = {'Conv2D': 1, 'Conv2D_1': 5, 'Conv2D_2': 8, 'Conv2D_3': 11, 'Conv2D_4': 14, 'Conv2D_5': 18, 'Conv2D_6': 21, 'Conv2D_7': 25, 'Conv2D_8': 28}

conv_layer = sorted(conv_dict.keys())#[1:]
top_bottom_pair_list = zip(conv_layer[1:],conv_layer[:-1])
outfile = 'models/decoder-param-0.pickle'
for top_bottom in top_bottom_pair_list:
    top_name,bottom_name = top_bottom
    print(top_name,bottom_name)
    outfile = modify_net(outfile,decoder_input_data,top_name,bottom_name)
print(outfile)
