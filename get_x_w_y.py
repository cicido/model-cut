# -*- coding:utf-8 -*-
import torch_net_param
import tensorflow as tf
import numpy as np

'''
input_data = None
with open(dataPickle,'rb') as f:
    print('load...')
    input_data = pickle.load(f)
    print(input_data.shape)   
'''

#计算原Y值(gtY)
# get_gtY与get_X用到的net_param_file不同
# get_gtY始终使用未修改的网络
def get_gtY(net_param_file,input_data,op_name):
    input_shape = input_data.shape
    nPicsPerBatch = input_shape[1]
    nBatches = input_shape[0]
    nPointsPerPics = 10
    nPointPerBatch = nPicsPerBatch * nPointsPerPics
    nAllPoints = nPointPerBatch * nBatches
    points_dict = {}
    conv_list = None
    
    with tf.Graph().as_default() as g, tf.Session(graph=g) as sess, tf.variable_scope(tf.get_variable_scope(), reuse=False) as scope:
        ph =  tf.placeholder(tf.float32,[None,None,None,None])
        c,conv_map = torch_net_param.construct_net(ph, g,net_param_file)
   
        for i in g.get_operation_by_name(op_name).inputs:
            if i.name.startswith(op_name):
                W_tensor = i

        W = sess.run(W_tensor)
        print("W.shape:",W.shape)
        W_shape = W.shape
    
        Y_tensor = g.get_tensor_by_name(op_name + ":0")

        idx = 0
        feats_Y = np.ndarray(shape=(nAllPoints,W_shape[3]))
        
        for batch_idx in range(nBatches):
            print("batch_indx:",batch_idx)
            Y = sess.run(Y_tensor,feed_dict={ph:input_data[batch_idx]})

            print("Y.shape:",Y.shape)
            Y_shape = Y.shape

            pixel_x = np.random.randint(0,Y_shape[1],nPointsPerPics)
            pixel_y = np.random.randint(0,Y_shape[2],nPointsPerPics)
            points_dict[(batch_idx,'x')] = pixel_x
            points_dict[(batch_idx,'y')] = pixel_y
            
            for point,x,y in zip(range(nPointsPerPics),pixel_x,pixel_y):
                i_from = idx + point*nPicsPerBatch
                try:
                    feats_Y[i_from:(i_from + nPicsPerBatch)] = Y[:,x,y,:].reshape((Y_shape[0],-1))
                except:
                    print("shape dismatched!")
                    raise Exception("shape dismatched!")
            idx += nPointPerBatch
        print("feats_Y.shape:",feats_Y.shape)
    return feats_Y, points_dict,W,conv_map

def get_X(net_param_file,input_data,op_name,points_dict):
    input_shape = input_data.shape
    nPicsPerBatch = input_shape[1]
    nBatches = input_shape[0]
    nPointsPerPics = 10
    nPointPerBatch = nPicsPerBatch * nPointsPerPics
    nAllPoints = nPointPerBatch * nBatches
    
    with tf.Graph().as_default() as g, tf.Session(graph=g) as sess, tf.variable_scope(tf.get_variable_scope(), reuse=False) as scope:
        ph = tf.placeholder(tf.float32,[None,None,None,None])
        c,conv_list = torch_net_param.construct_net(ph, g,net_param_file)
   
        for i in g.get_operation_by_name(op_name).inputs:
            if i.name.startswith(op_name):
                W_tensor = i
            else:
                X_tensor = i
                print(X_tensor)

        W = sess.run(W_tensor)
        print("W_shape:",W.shape)
        W_shape = W.shape

        idx = 0
        feats_X = np.ndarray(shape=(nAllPoints,W_shape[0],W_shape[1],W_shape[2]))
        
        for batch_idx in range(nBatches):
            print("batch_indx:",batch_idx)          
            X = sess.run(X_tensor,feed_dict={ph:input_data[batch_idx]}) 
            print("X.shape:",X.shape)
            pixel_x = points_dict[(batch_idx,'x')]
            pixel_y = points_dict[(batch_idx,'y')]
            for point,x,y in zip(range(nPointsPerPics),pixel_x,pixel_y):
                i_from = idx + point*nPicsPerBatch
                try:
                    x_start,x_end = x,x+W_shape[0]
                    y_start,y_end = y,y+W_shape[1]
                    feats_X[i_from:(i_from + nPicsPerBatch)] = X[:,x_start:x_end,y_start:y_end,:]
                except:
                    print("shape dismatched!")
                    raise Exception("shape dismatched!")
            idx += nPointPerBatch
    print("feats_X.shape",feats_X.shape)
    return feats_X