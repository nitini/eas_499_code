import os
import subprocess
import sys
import random
import pprint
import matplotlib
matplotlib.use('Agg')
import caffe
import time

def create_cv_partitions(fold_map, training_data_folder, k):
    p_classes = os.listdir(training_data_folder)
    for p in p_classes:
        p_dir = training_data_folder + '/' + p
        img_files = [p_dir + '/' + img for img in os.listdir(p_dir)]
        random.shuffle(img_files)
        fold_assignments = list(chunks(img_files, k))
        for f_a, i in zip(fold_assignments, range(k)):
            fold_map[i].extend(f_a)

def chunks(l, n):
    newn = int(len(l) / n)
    for i in xrange(0, n-1):
        yield l[i*newn:i*newn+newn]
    yield l[n*newn-newn:]

def create_training_and_holdout(fold_map, k, i):
    data = {'training': [], 'holdout':[]}
    for j in range(k):
        if j != i:
            data['training'].extend(fold_map[j])
        else:
            data['holdout'].extend(fold_map[j])
    return data

def prepare_data_for_lmdb_conversion(data):
    for key in data.keys():
        key_path = './cv_' + key
        out_txt = open('cv_' + key + '_classes.txt', 'wb')
        if os.path.isdir(key_path):
            subprocess.call('rm -rf ' + key_path, shell=True)
            subprocess.call('mkdir ' + key_path, shell=True)
        else:
            subprocess.call('mkdir ' + key_path, shell=True)
        copied_files = 0
        t0 = time.time()
        for img_file in data[key]:
            img_file_parts = img_file.split('/')
            class_label = img_file_parts[-1].split('_')[0]
            lmdb_img_file = key_path + '/' + img_file_parts[-1]
            cp_cmd = 'cp ' + img_file + ' ' + lmdb_img_file
            subprocess.call(cp_cmd, shell=True)
            copied_files += 1
            out_txt.write(img_file_parts[-1] + ' ' + class_label + '\n')
            if copied_files % 500 == 0:
                print "Copied over " + str(copied_files) + " files to create " + \
                        key + " fold"
        print "Copied over " + str(copied_files) + " total files"
        print "Took " + str(round(time.time() - t0, 2)) + " seconds to copy"

def create_lmdb(partitions):
    caffe_dir = '/usr/local/caffe/bin'
    convert_imageset = caffe_dir + '/convert_imageset.bin'
    for part in partitions:
        lmdb_dir = './cv_' + part + '_lmdb' 
        if os.path.isdir(lmdb_dir):
            subprocess.call('rm -rf ' + lmdb_dir, shell=True)
        part_dir = ' ./cv_' + part + '/ ' 
        part_txt_dir = './cv_' + part + '_classes.txt '
        convert_lmdb_cmd = convert_imageset + part_dir \
                           + part_txt_dir + lmdb_dir
        subprocess.call(convert_lmdb_cmd, shell=True)

def train_caffe_seaNet(seaNet_solver):
    caffe_dir = '/usr/local/caffe/bin'
    train_cmd = caffe_dir + '/caffe.bin train'
    solver_option = ' --solver=' + seaNet_solver
    train_cmd = train_cmd + solver_option
    t0 = time.time()
    subprocess.call(train_cmd, shell=True)
    print "Took " + str(round(time.time() - t0, 2)) + " seconds to train"

def compute_loss_and_acc():
    pass

def main():
    k = int(sys.argv[1])
    training_data_folder = sys.argv[2]
    fold_map = {}
    for i in range(k):
        fold_map[i] = []
    create_cv_partitions(fold_map, training_data_folder, k)
    for i in range(k):
        print "Training Net, Holdout fold is fold: " + str(i)
        data = create_training_and_holdout(fold_map, k, i)
        prepare_data_for_lmdb_conversion(data)
        create_lmdb(data.keys())
        break
        

if __name__ == '__main__':
    main()

