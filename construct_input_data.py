# -*- coding:utf-8 -*-
# construct data
import os
import tensorflow as tf
import numpy as np
import pickle
nPicsPerBatch = 10
resize = [512,512]
#dataDir = 'input/content/'
#dataPickle = "input/data.pickle"
def freeze_image_data(data_dir,output_file):
    imagelist = [os.path.join(data_dir,i) for i in os.listdir(data_dir) if os.path.isfile(data_dir + os.sep + i)]
    print("imagelist len:",len(imagelist))
    nBatches = int(len(imagelist) / nPicsPerBatch)
    nBatches = min(500,nBatches)
    print("nBatches:",nBatches)
    nPics = nPicsPerBatch*nBatches
    print(nPics)

    feats_shape = [nBatches,nPicsPerBatch] + resize + [3]
    print(feats_shape)

    feats_dict = np.ndarray(shape=feats_shape)

    with tf.Session() as sess:
        ph = tf.placeholder(tf.string)
        image = tf.image.decode_jpeg(tf.read_file(ph))

        image = tf.cast(image,tf.float32)
        image = tf.reverse(image, axis=[-1]) / 255.0  
        image = tf.image.resize_images(image, resize)

        for idx,filename in enumerate(imagelist[:nPics]):
            res = sess.run(image,feed_dict={ph:filename})
            print(idx,filename)
            a,b = divmod(idx,nPicsPerBatch)
            feats_dict[a,b] = res
    freeeze_data(feats_dict, output_file)
    
def freeze_data(input_data,output_file):
    with open(output_file,'wb') as f:
        print('pickle...')
        pickle.dump(input_data,f,protocol = 4)

def load_image_data(pickfile):
    fdict = None
    with open(pickfile,'rb') as f:
        print('load...')
        fdict = pickle.load(f)
    return fdict