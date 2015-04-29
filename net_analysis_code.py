"""
Created on Mon Apr 27 00:38:27 2015

@author: nitiniyer
"""

#%%
import numpy as np
import matplotlib
matplotlib.rcParams['backend'] = "Agg"
from matplotlib.backends.backend_pdf import PdfPages
import caffe
import lmdb
import sys
import time
import csv
import subprocess
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
    
x_axis = make_x_axis()
#%%
def process_net_output(in_file):
    test_acc = np.empty([1,100])
    test_loss = np.empty([1,100])
    train_loss = np.empty([1,100])
    input_file = open(in_file, 'rb')
    line = input_file.next()
    prev_line = line
    acc_count = 0
    loss_count = 0
    train_loss_count = 0
    while (True):
        try:
            if 'Test net output #0: accuracy' in line:
                if acc_count < 100:
                    test_acc[0,acc_count] = float(line.split(' ')[-1])
                acc_count += 1
            if 'Test net output #1: loss' in line:
                if loss_count < 100:
                    loss_piece = line.split('net output #1: loss = ')
                    test_loss[0,loss_count] = loss_piece[-1].strip().split(' ')[0]
                loss_count += 1
            if 'Train net output #0: loss = ' in line:
                prev_line_piece = prev_line.split('Iteration ')
                if int(prev_line_piece[-1].split(',')[0]) % 1000 == 0:
                    train_piece = line.split('net output #0: loss = ')[-1]
                    train_loss[0,train_loss_count] = float(train_piece.split(' ')[0])
                    train_loss_count += 1
            prev_line = line
            line = input_file.next()
        except StopIteration:
            break
    return {'train_loss': train_loss, 'test_loss': test_loss, 'test_acc': test_acc}
    

net_info_1 = process_net_output('./net_outputs/run_train_net.sh.o62072')
#%%
def plot_single_metric(y_vals, y_vals_2, y_vals_3):
    pp = PdfPages('train_v_test_001.pdf')
    x_axis = make_x_axis().tolist()[0]
    matplotlib.pyplot.clf()
    matplotlib.pyplot.xlabel('Iterations',fontsize=14)
    matplotlib.pyplot.title('Train v. Test Loss (LR = 0.001)',fontsize=16)
    matplotlib.pyplot.ylabel('Softmax Loss',fontsize=14)
    matplotlib.pyplot.plot(x_axis, y_vals.tolist()[0])
    matplotlib.pyplot.plot(x_axis, y_vals_2.tolist()[0])
    #matplotlib.pyplot.plot(x_axis, y_vals_3.tolist()[0])
    matplotlib.pyplot.legend(['Test Loss', 'Train Loss'], loc='middle right')
    pp.savefig()
    matplotlib.pyplot.show()
    pp.close()
    
plot_single_metric(net_info_001['test_loss'], net_info_001['train_loss'],net_info_001['test_acc'])

#%%
def class_accuracies():
    MODEL_FILE = './caffe_models/11_seaNet_deploy.prototxt'
    PRETRAINED = './caffe_models/seaNet_final_v11.caffemodel'
    mean_file = '../net_dev_data/train_all_48_mean.npy'
    lmdb_folder = './data/cv_holdout_lmdb'
    train_folder = '../train'
    classes = os.listdir(train_folder)
    classes.sort()
    seaNet = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
    caffe.set_mode_gpu()
    image_mean = np.load(mean_file)
    env = lmdb.open(lmdb_folder)
    txn = env.begin()
    cursor = txn.cursor()
    count = 0
    class_correct = np.empty([1,121])
    class_count = np.empty([1,121])
    predictions = []
    actuals = []
    for key, value in cursor:
        count += 1
        if count % 500 == 0:
            print 'Number of Images Processed: ' + str(count)
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value)
        label = datum.label
        actuals.append(int(label))
        class_count[0,label] += 1
        image = caffe.io.datum_to_array(datum)
        image = image.astype(np.uint8)
        image = image - image_mean
        image = image * 0.00390625
        result = seaNet.forward_all(data=np.array([image]))
        probs = result['prob'][0]
        predicted_class = probs.argmax()
        predictions.append(int(probs.argmax()))
        if predicted_class == label:
            class_correct[0,label] += 1
    class_accuracy = class_correct / class_count
    bottom_20 = heapq.nsmallest(20, range(len(class_accuracy.tolist()[0])), 
                             class_accuracy.tolist()[0].__getitem__)
    bottom_20_names = operator.itemgetter(*bottom_20)(classes)
    bottom_20_vals = operator.itemgetter(*bottom_20)(class_accuracy.tolist()[0])
    
    top_20 = heapq.nlargest(20, range(len(class_accuracy.tolist()[0])), 
                             class_accuracy.tolist()[0].__getitem__)
    top_20_names = operator.itemgetter(*top_20)(classes)
    top_20_vals = operator.itemgetter(*top_20)(class_accuracy.tolist()[0])
    return {'accuracies': class_accuracy, 'bottom_20_names':bottom_20_names,
    'bottom_20_vals':bottom_20_vals, 'top_20_names':top_20_names, 'top_20_vals':top_20_vals,
    'predictions': predictions, 'actuals': actuals}
model_accuracies = class_accuracies()

#%%
num_classes = 121
pp = PdfPages('seaNet_11_22500_acc_sort.pdf')
fig, ax = matplotlib.pyplot.subplots()
index = np.arange(num_classes)
matplotlib.pyplot.xlim([0,121])
rects = matplotlib.pyplot.bar(index, model_accuracies['accuracies'].tolist()[0])
matplotlib.pyplot.xlabel('Plankton Class')
matplotlib.pyplot.ylabel('% Correctly Classified')
matplotlib.pyplot.title('Class Accuracies on Validation Set (Sorted)')
pp.savefig()
pp.close()
#%%
def prediction_prob():
    MODEL_FILE = './caffe_models/11_seaNet_deploy.prototxt'
    PRETRAINED = './caffe_models/seaNet_final_v11.caffemodel'
    mean_file = '../net_dev_data/train_all_48_mean.npy'
    lmdb_folder = './data/cv_holdout_lmdb'
    train_folder = '../train'
    classes = os.listdir(train_folder)
    classes.sort()
    seaNet = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
    caffe.set_mode_gpu()
    image_mean = np.load(mean_file)
    env = lmdb.open(lmdb_folder)
    txn = env.begin()
    cursor = txn.cursor()
    count = 0
    prob_example = ""
    predictions = []
    actuals = []
    for key, value in cursor:
        count += 1
        if count % 500 == 0:
            print 'Number of Images Processed: ' + str(count)
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value)
        label = datum.label
        actuals.append(int(label))
        image = caffe.io.datum_to_array(datum)
        image = image.astype(np.uint8)
        image = image - image_mean
        image = image * 0.00390625
        result = seaNet.forward_all(data=np.array([image]))
        probs = result['prob'][0]
        predictions.append(int(probs.argmax()))
        print "Actual Class: " + str(label)
        print "Predicted Class: " + str(probs.argmax())
        prob_example = probs
        break
    return prob_example

prob_ex = prediction_prob()

#%%
num_classes = 121
pp = PdfPages('class_prob_example.pdf')
fig, ax = matplotlib.pyplot.subplots()
index = np.arange(num_classes)
matplotlib.pyplot.xlim([0,121])
matplotlib.pyplot.ylim([0,1])
rects = matplotlib.pyplot.bar(index, prob_ex)
#matplotlib.pyplot.plot(prob_ex)
matplotlib.pyplot.xlabel('Plankton Class')
matplotlib.pyplot.ylabel('Predicted Probability')
matplotlib.pyplot.title('Example of Net Class Probability Predictions')
pp.savefig()
pp.close()
#%%
def make_class_axis():
    x = np.arange(0,121)
    x_axis = np.empty([1,121])
    for x_val, i in zip(x, range(121)):
        x_axis[0,i] = x_val
    return x_axis


#%%
from sklearn.metrics import classification_report

print(classification_report(model_accuracies['actuals'], model_accuracies['predictions'], target_names=class_names))

f = open('classification_report.txt', 'wb')
print >> f, classification_report(model_accuracies['actuals'], model_accuracies['predictions'], target_names=class_names)
f.close()




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
    

num_models = 7
softmax_losses_all = (4.795791, 2.94217, 2.359601, 1.921511, 1.780583, 1.515769, 1.167927)
softmax_losses_cnn = (2.94217, 1.921511, 1.780583, 1.515769, 1.167927)
pp = PdfPages('model_losses.pdf')
fig, ax = matplotlib.pyplot.subplots()
index = np.arange(num_models)
bar_width = 0.4

rects = matplotlib.pyplot.bar(index, softmax_losses_all, bar_width)
ax.yaxis.set_ticks(np.arange(0,5.5,0.3))
matplotlib.pyplot.xlabel('Model')
matplotlib.pyplot.ylabel('Softmax Loss')
matplotlib.pyplot.title('Softmax Loss for Each Version of Net and Baselines')
matplotlib.pyplot.xticks(index + (bar_width / 2), ('Equal Prob.', 'CNN v1', 'RF v3', 'CNN v2', 'CNN v3', 'CNN v4', 'CNN Final'))
pp.savefig()
pp.close()

#%%
import numpy as np
from sklearn.metrics import confusion_matrix

class_names = os.listdir('../train')
class_names.sort()
# Split the data into a training set and a test set
# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=matplotlib.pyplot.cm.Blues):
    pp = PdfPages('confusion_matrix.pdf')
    matplotlib.pyplot.imshow(cm, interpolation='nearest', cmap=cmap)
    matplotlib.pyplot.title('Confusion Matrix for V11 Network')
    matplotlib.pyplot.colorbar()
    #tick_marks = np.arange(len(class_names))
    #matplotlib.pyplot.xticks(tick_marks, class_names, rotation=45)
    #matplotlib.pyplot.yticks(tick_marks, class_names)
    #plt.tight_layout()
    matplotlib.pyplot.ylabel('True label')
    matplotlib.pyplot.xlabel('Predicted label')
    pp.savefig()
    pp.close()


# Compute confusion matrix
cm = confusion_matrix(model_accuracies['actuals'], model_accuracies['predictions'])
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
matplotlib.pyplot.figure()
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
max_confusion = -1
max_index = []
matplotlib.pyplot.show()

for i in range(121):
    for j in range(121):
        if i !=j:
            if cm[i,j] > max_confusion:
                max_confusion = cm[i,j]
                max_index = [i,j]
print max_confusion
print max_index



