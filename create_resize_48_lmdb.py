import os
import subprocess
import sys

parent_dir = sys.argv[1]
copy_dir = sys.argv[2]
for class_dir in os.listdir(parent_dir):
    p_class_dir = parent_dir + '/' + class_dir
    for img in os.listdir(p_class_dir):
        img_path = p_class_dir + '/' + img
        copy_path = copy_dir + '/' + img
        cp_cmd = 'cp ' + img_path + ' ' + copy_path
        subprocess.call(cp_cmd, shell=True)

