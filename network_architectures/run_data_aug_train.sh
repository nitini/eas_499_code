#!/bin/sh
#$ -cwd
#$ -j y
#$ -m e -M nitini@wharton.upenn.edu
caffe train --solver=/home/nitini/eas_499_code/network_architectures/seaNet_data_aug_solver_all.prototxt