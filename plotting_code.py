"""
Created on Mon Apr 27 00:38:27 2015

@author: nitiniyer
"""

#%%
import numpy as np
import matplotlib
matplotlib.rcParams['backend'] = "GTK"
from matplotlib.backends.backend_pdf import PdfPages
#import caffe
#import lmdb
import sys
import numpy as np
import os
import heapq
import operator



#%%
def make_x_axis():
    x = np.arange(0,100000,1000)
    x_axis = np.empty([1,100])
    for x_val, i in zip(x, range(100)):
        x_axis[0,i] = x_val
    return x_axis
#%%
def make_data_aug_axis():
    x = np.arange(0,18000,1000)
    x_axis = np.empty([1,18])
    for x_val, i in zip(x, range(18)):
        x_axis[0,i] = x_val
    return x_axis
    
x_axis = make_x_axis()
x_data_aug_axis = make_data_aug_axis()
#%%
import PIL
import PIL.Image

im = PIL.Image.open('./v14_table.jpg')

newfilename = './v14_table.pdf'
PIL.Image.Image.save(im, newfilename, "PDF", resoultion = 100.0)

#%%
def process_net_output_old(in_file, iters):
    test_acc = np.empty([1,iters])
    test_loss = np.empty([1,iters])
    train_loss = np.empty([1,iters])
    input_file = open(in_file, 'rb')
    line = input_file.next()
    prev_line = line
    acc_count = 0
    loss_count = 0
    train_loss_count = 0
    while (True):
        try:
            if 'Test net output #0: accuracy' in line:
                if acc_count < iters:
                    test_acc[0,acc_count] = float(line.split(' ')[-1])
                acc_count += 1
            if 'Test net output #1: loss' in line:
                if loss_count < iters:
                    loss_piece = line.split('net output #1: loss = ')
                    test_loss[0,loss_count] = loss_piece[-1].strip().split(' ')[0]
                loss_count += 1
            if 'Train net output #0: loss = ' in line:
                prev_line_piece = prev_line.split('Iteration ')
                if int(prev_line_piece[-1].split(',')[0]) % 1000 == 0:
                    train_piece_loss = line.split('net output #0: loss = ')[-1]
                    train_loss[0,train_loss_count] = float(train_piece_loss.split(' ')[0])
                    train_loss_count += 1
            prev_line = line
            line = input_file.next()
        except StopIteration:
            break
    return {'train_loss': train_loss, 'test_loss': test_loss, 'test_acc': test_acc}
    

net_info_11 = process_net_output_old('./net_outputs/v5_output.txt', 100)


#%%
def process_net_output(in_file, iters):
    test_acc = np.empty([1,iters])
    test_loss = np.empty([1,iters])
    train_loss = np.empty([1,iters])
    train_acc = np.empty([1,iters])
    input_file = open(in_file, 'rb')
    line = input_file.next()
    prev_line = line
    acc_count = 0
    loss_count = 0
    train_loss_count = 0
    train_acc_count = 0
    while (True):
        try:
            if 'Test net output #0: accuracy' in line:
                if acc_count < iters:
                    test_acc[0,acc_count] = float(line.split(' ')[-1])
                acc_count += 1
            if 'Test net output #1: loss' in line:
                if loss_count < iters:
                    loss_piece = line.split('net output #1: loss = ')
                    test_loss[0,loss_count] = loss_piece[-1].strip().split(' ')[0]
                loss_count += 1
            if 'Train net output #0: accuracy = ' in line:
                prev_line_piece = prev_line.split('Iteration ')
                if int(prev_line_piece[-1].split(',')[0]) % 1000 == 0:
                    train_piece_acc = line.split('net output #0: accuracy = ')[-1]
                    train_acc[0,train_acc_count] = float(train_piece_acc.split(' ')[0])
                    line = input_file.next()
                    train_piece_loss = line.split('net output #1: loss = ')[-1]
                    train_loss[0,train_loss_count] = float(train_piece_loss.split(' ')[0])
                    train_loss_count += 1
                    train_acc_count += 1
            prev_line = line
            line = input_file.next()
        except StopIteration:
            break
    return {'train_loss': train_loss,'train_acc': train_acc, 'test_loss': test_loss, 'test_acc': test_acc}
    

net_info_11 = process_net_output('./net_outputs/v11_output.txt', 100)

#%%
def plot_single_metric(y_vals, y_vals_2, y_vals_3, y_vals_4, y_vals_5):
    pp = PdfPages('all_test_accc.pdf')
    x_axis = make_x_axis().tolist()[0]
    #x_axis = make_data_aug_axis().tolist()[0]
    matplotlib.pyplot.clf()
    matplotlib.pyplot.xlabel('Iterations', fontsize=14)
    matplotlib.pyplot.title('Versions of CNN Acc.',fontsize=16)
    matplotlib.pyplot.ylabel('Accuracy',fontsize=14)
    matplotlib.pyplot.plot(x_axis, y_vals.tolist()[0])
    matplotlib.pyplot.plot(x_axis, y_vals_2.tolist()[0])
    matplotlib.pyplot.plot(x_axis, y_vals_3.tolist()[0])
    matplotlib.pyplot.plot(x_axis, y_vals_4.tolist()[0])
    matplotlib.pyplot.plot(x_axis, y_vals_5.tolist()[0])
    matplotlib.pyplot.legend(['V2.', 'V3', 'V4', 'V11', 'V14'], loc='upper right')
    pp.savefig()
    matplotlib.pyplot.show()
    pp.close()
    
plot_single_metric(net_info_2['test_acc'], net_info_3['train_acc'],net_info_4['test_loss'], net_info_11['test_loss'], net_info_14['test_loss'])

#%%
def class_accuracies():
    MODEL_FILE = sys.argv[1]
    PRETRAINED = sys.argv[2]
    mean_file = sys.argv[3]
    lmdb_folder = sys.argv[4]
    train_folder = sys.argv[5]
    seaNet = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
    caffe.set_mode_gpu()
    image_mean = np.load(mean_file)
    env = lmdb.open(lmdb_folder)
    txn = env.begin()
    cursor = txn.cursor()
    count = 0
    class_correct = np.empty([1,121])
    class_count = np.empty([1,121])
    for key, value in cursor:
        count += 1
        if count % 500 == 0:
            print 'Number of Images Processed: ' + str(count)
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value)
        label = datum.label
        class_count[0,label] += 1
        image = caffe.io.datum_to_array(datum)
        image = image.astype(np.uint8)
        image = image - image_mean
        image = image * 0.00390625
        result = seaNet.forward_all(data=np.array([image]))
        probs = result['prob'][0]
        predicted_class = probs.argmax()
        if predicted_class == label:
            class_correct[label] += 1
    class_accuracy = class_correct / class_count
    print aclass_accuracy
#%%
def make_class_axis():
    x = np.arange(0,121)
    x_axis = np.empty([1,121])
    for x_val, i in zip(x, range(121)):
        x_axis[0,i] = x_val
    return x_axis







#%%
def make_class_distribution_bar_chart():
    train_folder = '../train'
    classes = os.listdir(train_folder)
    classes.sort()
    class_counts = np.empty([1,121])
    for p_class, i in zip(classes, range(121)):
        num_imgs = len(os.listdir(train_folder + '/' + p_class))
        class_counts[0,i] = num_imgs
    top_10 = heapq.nsmallest(10, range(len(class_counts.tolist()[0])), 
                            class_counts.tolist()[0].__getitem__)
    print operator.itemgetter(*top_10)(classes)
    print operator.itemgetter(*top_10)(class_counts.tolist()[0])
    return class_counts

plankton_class_counts = make_class_distribution_bar_chart()
num_classes = 121
pp = PdfPages('plankton_class_distribution.pdf')
fig, ax = matplotlib.pyplot.subplots()
index = np.arange(num_classes)
matplotlib.pyplot.xlim([0,121])
rects = matplotlib.pyplot.bar(index, plankton_class_counts.tolist()[0])
matplotlib.pyplot.xlabel('Plankton Class Index Value')
matplotlib.pyplot.ylabel('Count of Training Images')
matplotlib.pyplot.title('Class Distribution for Images')
pp.savefig()
pp.close()

"""
('trichodesmium_puff', 'chaetognath_other', 'copepod_cyclopoid_oithona_eggs',
 'protist_other', 'detritus_other', 'copepod_cyclopoid_oithona', 'acantharia_protist', 
 'chaetognath_non_sagitta', 'trichodesmium_bowtie', 'hydromedusae_solmaris')
(1979.0, 1934.0, 1189.0, 1172.0, 914.0, 899.0, 889.0, 815.0, 708.0, 703.0)


('hydromedusae_haliscera_small_sideview', 'fish_larvae_deep_body', 'heteropod', 
'hydromedusae_other', 'acantharia_protist_big_center', 'pteropod_theco_dev_seq', 
'ephyra', 'hydromedusae_typeE', 'invertebrate_larvae_other_A', 
'appendicularian_fritillaridae')
(9.0, 10.0, 10.0, 12.0, 13.0, 13.0, 14.0, 14.0, 14.0, 16.0)

"""
#%%
# Make Plot for Model Softmax Loss
    

num_models = 6
softmax_losses_all = (4.795791, 2.359601, 1.921511, 1.780583, 1.515769, 1.167927)
softmax_losses_cnn = (2.94217, 1.921511, 1.780583, 1.515769, 1.167927)
pp = PdfPages('model_losses.pdf')
fig, ax = matplotlib.pyplot.subplots()
index = np.arange(num_models)
bar_width = 0.4

rects = matplotlib.pyplot.bar(index, softmax_losses_all, bar_width)
ax.yaxis.set_ticks(np.arange(0,5.5,0.3))
matplotlib.pyplot.xlabel('Model')
matplotlib.pyplot.ylabel('Multiclass Log Loss')
matplotlib.pyplot.title('Multiclass Loss for Each Version of Net and Baselines')
matplotlib.pyplot.xticks(index + (bar_width / 2), ('Equal Prob.', 'RF V3', 'CNN V2', 'CNN V3', 'CNN V4', 'CNN V11'))
pp.savefig()
pp.close()

#%%

train_folder = '../train'

classes = os.listdir(train_folder)
size = []
for p_class in classes:
    size.append(len(os.listdir(train_folder + '/' + p_class)))

size.sort()
print size
    