import caffe
import numpy as np
import sys

mean_file = sys.argv[1]
mean_out_file = sys.argv[2]

blob = caffe.proto.caffe_pb2.BlobProto()
data = open(mean_file, 'rb').read()
blob.ParseFromString(data)
npy_arr = np.array(caffe.io.blobproto_to_array(blob))
print npy_arr[0]
np.save(mean_out_file, npy_arr[0])
