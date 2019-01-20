#date:20181113
#anthor:weidi
#aim:change the convolution kernel in the *.caffemodel file
# -*- coding:utf-8 -*- 
import sys
sys.path.append('/home/dell/WD/caffe-master/python')
import caffe

import os
import numpy as np 
import h5py
import scipy.io as scio

##############################################################################################
#################################      Setting       #########################################
#### Mat_file 
# Each 4-d matrix represents a convolution kernel
# Each 4-d matrix shape: [ N2 x N1 x W x H ]
# N2: Channels of output feature, Num of conv kernels;
# N1: Channels of input feature;
net_deploy = './deploy_lenet.prototxt'
net_caffemodel = './lenet_iter_10.caffemodel'
net_caffemodel_new = './lenet_iter_10_new.caffemodel'
mat_file = './new_conv.mat'
change_params = ['conv1','conv2']


##############################################################################################
################################ check if new file exist #####################################
if os.path.exists(net_caffemodel_new):
    print '\nError:{} already exists, please rename it!'.format(net_caffemodel_new)
else:

##############################################################################################
#################################   Read caffemodel  #########################################
    caffe.set_mode_gpu 
    net0 = caffe.Net(net_deploy,net_caffemodel,caffe.TEST) 
    keys0 = net0.params.keys() 
    print '\n########  All Layers name:   ############\n',keys0


##############################################################################################
#################################   load Matfile    ##########################################
    print '\n########  Load Mat File...   ############'
    mat_data = h5py.File(mat_file)
    new_conv = np.empty((len(change_params)),dtype=object)
    i=0    
    for params in change_params:
        #new_conv[i] = mat_data[params]
        new_conv[i] = np.transpose(mat_data[params])
        print 'mat_file:{}-shape:'.format(params),new_conv[i].shape
        i=i+1


##############################################################################################
#################################   Saving New Model    ######################################
    print '\n######## Saving New Model... ############'
    print 'changed_layers:',change_params
    net1 = net0
    i=0
    for params in change_params:
        #print params
        print '{}_old_shape:'.format(params),net1.params[params][0].data[:].shape
        net1.params[params][0].data[:] = new_conv[i]
        print '{}_new_shape:'.format(params),net1.params[params][0].data[:].shape
        i=i+1

    net1.save(net_caffemodel_new)
    print '\n########        Done!       #############'






