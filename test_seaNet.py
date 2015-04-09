import sys
import caffe
import numpy as np

MODEL_FILE = sys.argv[1]
PRETRAINED = sys.argv[2]
final_layer = sys.argv[3]
total_files = int(sys.argv[4])
seaNet = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
caffe.set_mode_gpu()
preds = seaNet.blobs[final_layer].data
batch_size = np.shape(preds)[0]
correct = 0
count = 0
keep_going = True
while(keep_going):
    result = seaNet.forward()
    print seaNet.blobs['data'].data[0]
    labels = seaNet.blobs['label'].data
    preds = seaNet.blobs[final_layer].data
    for pred, label in zip(preds, labels):
        print "Generating Prediction for Image Number: " + str(count)
        count += 1
        prob = np.exp(pred) / np.sum(np.exp(pred))
        if int(prob.argmax()) == int(label):
            correct += 1
        if count == total_files:
            keep_going = False
            break
print count
print correct

