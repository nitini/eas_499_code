import subprocess
import os

if os.path.isdir('./resize_48_test'):
    subprocess.call('rm -rf ./resize_48_test', shell=True)
    subprocess.call('mkdir ./resize_48_test', shell=True)
else:
    subprocess.call('mkdir ./resize_48_test', shell=True)

label_file = open('fake_test_labels.txt','wb')
for img in os.listdir('./test'):
    print img
    subprocess.call('convert ' + './test/' + img + ' -resize 48x48\! '\
            + './resize_48_test/size_48_' + img, shell=True)
    label_file.write('size_48_' + img + ' ' + img.split('.')[0] + '\n')
label_file.close()

