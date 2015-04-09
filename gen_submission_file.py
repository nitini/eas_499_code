import caffe
import lmdb
import sys
import time
import csv
import numpy as np
import os

def setup_submission_file(train_folder, file_name):
    header = ['image']
    p_classes = os.listdir(train_folder)
    p_classes.sort()
    for p_class in p_classes:
        header.append(p_class)
    submission_file = open(file_name, 'wb')
    submission_writer = csv.writer(submission_file)
    submission_writer.writerow(header)
    submission_file.close()

def main():
    MODEL_FILE = sys.argv[1]
    PRETRAINED = sys.argv[2]
    mean_file = sys.argv[3]
    lmdb_folder = sys.argv[4]
    train_folder = sys.argv[5]
    seaNet = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
    caffe.set_mode_gpu()
    image_mean = np.load(mean_file)
    file_name = 'seaNet_submission_' + ('%0.f' % time.time()) + '.csv'
    setup_submission_file(train_folder, file_name)
    submission_file = open(file_name, 'a')
    submission_writer = csv.writer(submission_file)
    env = lmdb.open(lmdb_folder)
    txn = env.begin()
    cursor = txn.cursor()
    count = 0
    for key, value in cursor:
        print "Number of Images Processed: " + str(count)
        count += 1
        print key
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value)
        label = datum.label
        image = caffe.io.datum_to_array(datum)
        image = image.astype(np.uint8)
        image = image - image_mean
        image = image * 0.00390625
        result = seaNet.forward_all(data=np.array([image]))
        probs = result['prob'][0]
        img_row = [ '_'.join(key.split('_')[1:])]
        img_row.extend(probs)
        submission_writer.writerow(img_row)
    submission_file.close()


if __name__ == "__main__":
    main()

