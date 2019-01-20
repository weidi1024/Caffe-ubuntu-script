#date:20181113
#anthor:weidi
#aim:read *.caffemodel file and save to *.mat_file


import sys
sys.path.append('/home/dell/WD/caffe-master/python')
import scipy.io as sio
import numpy as np

import caffe
from caffe.proto import caffe_pb2

net_param = caffe_pb2.NetParameter()

def readmodel(modelpath,matpath):
    net_str = open(modelpath, 'r').read()
    net_param.ParseFromString(net_str)
    ls = len(net_param.layer)
    params = np.zeros((ls,3),dtype=np.object);
    for i in xrange(ls):
           name = net_param.layer[i].name
           blobs = net_param.layer[i].blobs
           lt = len(blobs)
           if lt==0 :
              weights = [] 
              bias = []
           elif lt==1 :
              d = blobs[0].shape.dim
              wdata = blobs[0].data
              weights = np.reshape(wdata,d)          
              bias = []
           else:
              d = blobs[0].shape.dim
              wdata = blobs[0].data
              weights = np.reshape(wdata,d)          
              bias = blobs[1].data
           params[i] = [name, weights, bias]
    sio.savemat(matpath, {'bnsc':params})
              
if __name__ == '__main__':

        modelfile0 = './lenet_iter_10.caffemodel'
        matfile0 = './lenet_iter_10.mat'

        modelfile1 = './lenet_iter_10_new.caffemodel'
        matfile1 = './lenet_iter_10_new.mat'

        readmodel(modelfile0,matfile0)
        readmodel(modelfile1,matfile1)
