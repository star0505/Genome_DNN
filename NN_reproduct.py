# coding: utf-8

import numpy as np
import random
import pdb
import tensorflow as tf
from tqdm import tqdm
import ConfigParser
from sklearn import metrics

config = ConfigParser.RawConfigParser()
config.read('config.cfg')

data_dir_rna = config.get('Parameter', 'data_dir_rna')
data_dir_beta = config.get('Parameter', 'data_dir_beta')

def file_read(data_dir):
	f = open(data_dir, 'r')
	lines = f.readlines()
	feature = tuple(lines[0].split('\t')[1:])
	samples = []
	data = np.zeros([100,len(feature)])
	for i, line in enumerate(lines):
		if i != 0:
			samples.append(line.split('\t')[0])
			data[i-1,:] = line.split('\t')[1:]
	#print data_dir
	f.close()
	return data, feature, samples

rna_data, rna_feature, samples = file_read(data_dir_rna)
beta_data, beta_feature, _ = file_read(data_dir_beta)
what_data_use = config.get('Parameter', 'what_data_use')
if what_data_use == 'rna':
	data = rna_data
	feature = rna_feature
else:
	if what_data_use == 'beta':
		data = beta_data
		feature = beta_feature
	else: 
		data = np.column_stack((rna_data,beta_data))
		feature = rna_feature + beta_feature

label = np.zeros(100)
for i,sample in enumerate(samples):
    if 'sdpc' in sample:
        label[i] = 0
    else:
        label[i] = 1

idx = range(len(data))
np.random.shuffle(idx)
data_x = data[idx].tolist()
data_y = label[idx].tolist()


# # Model


N, itr, lr, train_log, d, test_ratio, l2_regularizer_use = config.getint('Parameter', 'batch_size'), config.getint('Parameter','iteration'), config.getfloat('Parameter', 'learning_rate'), config.get('Parameter', 'train_dir'), config.getint('Parameter', 'hidden_dim'), config.getfloat('Parameter', 'test_ratio'), config.getboolean('Parameter', 'l2_regularizer_use')
print "Batch Size: ", N
print "Learning rate: ", lr

g_step = tf.Variable(0)
lr = tf.train.exponential_decay(lr, g_step, 100, 0.98)
x = tf.placeholder(tf.float32, [None, len(feature)],name="x")
y = tf.placeholder(tf.int32, [None], name="y")


W = tf.Variable(tf.zeros([len(feature),1]), name='W1')
b = tf.Variable(tf.zeros(1), name="b")
l2_regularization = tf.nn.l2_loss(W) + tf.nn.l2_loss(b)
y_ = tf.matmul(x, W) + b #[N,1]
y_ = tf.reshape(y_,[-1])

prediction = tf.greater(tf.sigmoid(y_), 0.5, name="prediction")
correct = tf.equal(prediction, tf.equal(y, True))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
if l2_regularizer_use == 1:
	loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(y, tf.float32),logits=y_) \
								+ l2_regularization
else: loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(y, tf.float32), logits=y_)
loss_mean = tf.reduce_mean(loss)
train_step = tf.train.RMSPropOptimizer(lr, 0.9).minimize(loss)

#Summary
tf.summary.scalar('loss', loss_mean)
tf.summary.scalar('accuracy', accuracy)
merged = tf.summary.merge_all()

saver = tf.train.Saver()
init = tf.global_variables_initializer()
sess = tf.Session()
train_writer = tf.summary.FileWriter(train_log, sess.graph)
sess.run(init)

def cross_validation(x,y,portion,test_ratio):
    t1 = test_ratio*portion
    t2 = test_ratio*(portion+1)
    train_x = np.array(x[:int(len(x)*t1)] + x[int(len(x)*(t2)):])
    train_y = np.array(y[:int(len(x)*t1)] + y[int(len(x)*(t2)):])
    test_x = np.array(x[int(len(x)*t1):int(len(x)*(t2))])
    test_y = np.array(y[int(len(x)*t1):int(len(x)*(t2))])
    return train_x, train_y, test_x, test_y

def random_batch(train_data, train_label, batch_size):
    idx = np.array(range(len(train_data)))
    np.random.shuffle(idx)
    x = np.array(train_data[idx])
    y = np.array(train_label[idx])
    return x[:batch_size], y[:batch_size]

#train_x, test_x = data_x[:int(len(data_x)*0.8)], data_x[int(len(data_x)*0.8):]
#train_y, test_y = label_y[:int(len(label_y)*0.8)], label_y[int(len(label_y)*0.8):]
total_test_acc = []
total_test_auc = []
save_result_dir = config.get('Parameter', 'save_result_dir')
for k in range(int(1/test_ratio)):
	with open(save_result_dir, 'a') as f:
            f.write(str(k))
            f.write(' fold')
            f.write('\n')
	sess.run(init)
	train_x, train_y, test_x, test_y = cross_validation(data_x,data_y,k,test_ratio)
	#TRAIN
	for i in range(itr):
		batch_x, batch_y = random_batch(train_x, train_y, N)
		_, acc, loss, train_sum, weight = sess.run([train_step, accuracy, loss_mean, merged, W],\
															feed_dict={x: batch_x, y: batch_y})
		#if i%100 == 0:
		#	train_writer.add_summary(train_sum, i)
		#	print "batch_time: " , i , "[*] Accuracy: ", acc
		#	print "Max: ", np.max(weight), "Min: ", np.min(weight) 
		#	print "feature(max): ", feature[np.argmax(weight)]
		#	print "feature(min): ", feature[np.argmin(weight)]
	max_idx = np.argsort(weight, axis=0)
        for j in range(5):
            print float(weight[int(max_idx[j])]), feature[int(max_idx[j])] 
            print float(weight[int(max_idx[-(j+1)])]), feature[int(max_idx[-(j+1)])]
            with open(save_result_dir, 'a') as f:
                f.write(str(float(weight[int(max_idx[j])])))
                f.write('\t')
                f.write(feature[int(max_idx[j])])
                f.write('\t')
                f.write(str(float(weight[int(max_idx[-(j+1)])])))
                f.write('\t')
                f.write(feature[int(max_idx[-(j+1)])])
                f.write('\n')
        #TEST
	test_acc, test_loss, test_pred, label = sess.run([accuracy, loss_mean, prediction, y], \
														feed_dict={x: test_x, y: test_y})
	fpr, tpr, thresholds = metrics.roc_curve(label, test_pred, pos_label=1)
	test_auc = metrics.auc(fpr, tpr)
	#print "[*] Fold", k ,"Test Accuracy: ", test_acc , ", loss: ", test_loss, "\n" 
	total_test_acc.append(test_acc)
	total_test_auc.append(test_auc)

total_acc = sum(total_test_acc)/len(total_test_acc)
total_auc = sum(total_test_auc)/len(total_test_auc)
print(total_acc)
print(total_auc)
with open(save_result_dir,'a') as f:
	f.write(str(total_acc))
	f.write('\t')
	f.write(str(total_auc))
	f.write('\n')
