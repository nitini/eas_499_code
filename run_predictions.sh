#!/bin/sh
#$ -cwd
#$ -j y
#$ -m e -M nitini@wharton.upenn.edu
python /home/nitini/eas_499_code/gen_submission_file.py /home/nitini/eas_499_code/network_architectures/11_seaNet_deploy.prototxt /home/nitini/eas_499_code/network_architectures/seaNet_11_22500.caffemodel /home/nitini/eas_499_code/network_architectures/train_all_48_mean.npy /home/nitini/data_files/test_all_48_lmdb /home/nitini/data_files/train
