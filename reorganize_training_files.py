import os
import subprocess
import sys

given_folder = sys.argv[1]
all_files_folder = sys.argv[2]
if os.path.isdir(all_files_folder):
    subprocess.call('rm -rf ' + all_files_folder, shell=True)
    subprocess.call('mkdir ' + all_files_folder, shell=True)
else:
    subprocess.call('mkdir ' + all_files_folder, shell=True)

for folder in os.listdir(given_folder):
    for img in os.listdir(given_folder + '/' + folder):
        file_path = given_folder + '/' + folder + '/' + img
        cp_cmd = 'cp ' + file_path + ' ' + all_files_folder + '/' + img
        subprocess.call(cp_cmd, shell=True)

