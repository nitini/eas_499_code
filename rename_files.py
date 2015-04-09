# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 18:11:34 2015

@author: nitiniyer
"""

import sys
import os
import subprocess

def relabel_classes(root_folder, given_folder):
    if os.path.isdir(given_folder):
        subprocess.call('rm -rf ' + given_folder, shell=True)
        subprocess.call('mkdir ' + given_folder, shell=True)
    else:
        subprocess.call('mkdir ' + given_folder, shell=True)
    class_labels = os.listdir(root_folder)
    class_labels.sort()

    class_num = 0
    count = 0
    for p_class in class_labels:
        for img in os.listdir(root_folder + '/' + p_class):
            print count
            print p_class
            print "Old: " + img
            img_path = root_folder + '/' + p_class + '/' + img
            img_parts = img.split('_')
            new_img = str(class_num)
            for part in img_parts[1:]:
                new_img = new_img + '_' + part
            print "New: " + new_img
            cp_cmd = 'cp ' + img_path + ' ' + given_folder + '/' + new_img
            subprocess.call(cp_cmd, shell=True)
            count += 1
        class_num += 1
        
def relabel_classes_keep_folders(root_folder, given_folder):
    if os.path.isdir(given_folder):
        subprocess.call('rm -rf ' + given_folder, shell=True)
        subprocess.call('mkdir ' + given_folder, shell=True)
    else:
        subprocess.call('mkdir ' + given_folder, shell=True)
    class_labels = os.listdir(root_folder)
    class_labels.sort()
    class_num = 0
    count = 0
    for p_class in class_labels:
        subprocess.call('mkdir ' + given_folder + '/' + p_class, shell=True)
        for img in os.listdir(root_folder + '/' + p_class):
            print count
            print p_class
            print "Old: " + img
            img_path = root_folder + '/' + p_class + '/' + img
            img_parts = img.split('_')
            new_img = str(class_num)
            for part in img_parts[1:]:
                new_img = new_img + '_' + part
            print "New: " + new_img
            file_path = given_folder + '/' + p_class
            cp_cmd = 'cp ' + img_path + ' ' + file_path + '/' + new_img
            subprocess.call(cp_cmd, shell=True)
            count += 1
        class_num += 1

    

def resize_test_files(root_folder, given_folder):
    count = 0
    if os.path.isdir(given_folder):
        subprocess.call('rm -rf ' + given_folder, shell=True)
        subprocess.call('mkdir ' + given_folder, shell=True)
    else:
        subprocess.call('mkdir ' + given_folder, shell=True)
    for img in os.listdir(root_folder):
        print count
        img_path = root_folder + '/' + img
        dest_path = given_folder + '/' + img
        dims = ' -resize 48x48\! '
        subprocess.call('convert ' + img_path + dims + dest_path, shell=True)
        count+= 1


def main():
    root_folder = sys.argv[1]
    given_folder = sys.argv[2]
    resize_test_files(root_folder, given_folder)

if __name__ == "__main__":
    main()
