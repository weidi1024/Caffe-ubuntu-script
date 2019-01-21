#there has a bug ,which when channel > 1 ,the created lmdb file is wrong
#to be continue...

#date:20180907
#anthor:weidi
#aim:creat lmdb file from mat file


caffe_root = '/home/dell/WD/caffe-master/'   #change it to your caffe path
import sys
sys.path.append(caffe_root+'python')
import numpy as np
import lmdb
import caffe
import h5py
import scipy.io as scio

##################################################################################################
##### READ ME!  
##### 1. mat : train_data = [N,C,W,H],test_data = [N,C,W,H],train_label=[N,1],test_label=[N,1]
##### 2. train data need to be shuffled before it
##################################################################################################

##################
# load mat
##################
print('load mat...')
mat_data = h5py.File('./data2.mat')    ## change it!
#mat_data = scio.loadmat('./data2.mat')  ## no need to trainspose



# load train data
train_data = mat_data['train_data']
train_data = np.transpose(train_data)
print 'train.shape:',train_data.shape
train_type = train_data.dtype
print 'train.type:',train_type
train_label = mat_data['train_label']
train_label = np.transpose(train_label)

# load test data
test_data = mat_data['test_data']
test_data = np.transpose(test_data)
test_label = mat_data['test_label']
test_label = np.transpose(test_label)
print 'test.shape:',test_data.shape
test_type = test_data.dtype
print 'test.type:',test_type
print('load done!\n')


##################
# creat train lmdb
##################
print('creating train lmdb...')
train_map_shape = train_data.shape
train_num = train_map_shape[0]
train_map_size = train_map_shape[0]*train_map_shape[1]*train_map_shape[2]*train_map_shape[3]*10

print('train_num:'),
print(train_map_shape[0])
print('train_data_map_size:'),
print(train_map_size)
env_train = lmdb.open('train_lmdb', map_size=train_map_size)
with env_train.begin(write=True) as txn_train:
    for i in range(train_num):
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = train_data.shape[1]
        datum.height = train_data.shape[2]
        datum.width = train_data.shape[3]
        if train_type == np.int:
             datum.data = train_data[i].tobytes()  # or .tostring() if numpy < 1.9
        elif train_type == np.float:
              datum.float_data.extend(train_data[i].flat)
        datum.label = int(train_label[i])
        str_id = '{:08}'.format(i)
        txn_train.put(str_id.encode('ascii'), datum.SerializeToString())
print('train lmdb done!\n')


##################
# creat test lmdb
##################
print('creating test lmdb...')
test_map_shape = test_data.shape
test_num = test_map_shape[0]
test_map_size = test_map_shape[0]*test_map_shape[1]*test_map_shape[2]*test_map_shape[3]*10
print('test_num:'),
print(test_map_shape[0])
print('test_data_map_size:'),
print(test_map_size)
env_test = lmdb.open('test_lmdb', map_size=test_map_size)
with env_test.begin(write=True) as txn_test:
    for i in range(test_num):
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = test_data.shape[1]
        datum.height = test_data.shape[2]
        datum.width = test_data.shape[3]
        if test_type == np.int:
             datum.data = test_data[i].tobytes()  # or .tostring() if numpy < 1.9
        elif test_type == np.float:
              datum.float_data.extend(test_data[i].flat)
        datum.label = int(test_label[i])
        str_id = '{:08}'.format(i)
        txn_test.put(str_id.encode('ascii'), datum.SerializeToString())
print('test lmdb done!')
