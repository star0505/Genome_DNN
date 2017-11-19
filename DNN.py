# coding: utf-8

import numpy as np
import random
import pdb
import tensorflow as tf
from tqdm import tqdm
import ConfigParser
from sklearn import metrics
from tqdm import tqdm

config = ConfigParser.RawConfigParser()
config.read('config.cfg')

data_dir_rna = config.get('Parameter', 'data_dir_rna')
data_dir_beta = config.get('Parameter', 'data_dir_beta')
data_dir_both = config.get('Parameter', 'data_dir_both')

def file_read(data_dir):
	f = open(data_dir, 'r')
	lines = f.readlines()
	feature = tuple(lines[0].split('\t')[1:])
	samples = []
	data = np.zeros([len(lines)-1,len(feature)])
	for i, line in enumerate(lines):
		if i != 0:
			samples.append(line.split('\t')[0])
			data[i-1,:] = line.split('\t')[1:]
	print data_dir
	f.close()
	return data, feature, samples

rna_data, rna_feature, samples = file_read(data_dir_rna)
beta_data, beta_feature, _ = file_read(data_dir_beta)
both_data, both_feature, _ = file_read(data_dir_both)
what_data_use = config.get('Parameter', 'what_data_use')

if what_data_use == 'rna':
	data = rna_data
	feature = rna_feature
else:
	if what_data_use == 'beta':
		data = beta_data
		feature = beta_feature
	else: 
		data = both_data
		feature = both_feature

label = np.zeros(len(data))
for i,sample in enumerate(samples):
    if 'msdp' in sample:
        label[i] = 1
    else:
        label[i] = 0

print("# of samples:", len(samples))
print("Dimension of data:", data.shape)
print("size of label:", len(label))
print("# of MDD:", len(np.argwhere(label==0)))
print("# of Normal:", len(np.nonzero(label)[0]))

#idx = range(len(data))
#np.random.shuffle(idx)


# # Model


N, itr, lr, train_log, d, test_ratio, l2_regularizer_use = config.getint('Parameter', 'batch_size'), config.getint('Parameter','iteration'), config.getfloat('Parameter', 'learning_rate'), config.get('Parameter', 'train_dir'), config.getint('Parameter', 'hidden_dim'), config.getfloat('Parameter', 'test_ratio'), config.getboolean('Parameter', 'l2_regularizer_use')
print "Batch Size: ", N
print "Learning rate: ", lr

g_step = tf.Variable(0)
lr = tf.train.exponential_decay(lr, g_step, 100, 0.98)
x = tf.placeholder(tf.float32, [None, len(feature)],name="x")
y = tf.placeholder(tf.int32, [None], name="y")

def multi_NN(feature, x, hidden_d):
    W1 = tf.Variable(tf.zeros([len(feature),d]), name='W1')
    b1 = tf.Variable(tf.zeros(d), name="b1")
    y_1 = tf.matmul(x, W1) + b1

    W2 = tf.Variable(tf.zeros([hidden_d, 1]), name='W1')
    b2 = tf.Variable(tf.zeros(1), name='b2')
    y_2 = tf.matmul(y_1, W2) + b2
    l2_regularization = tf.nn.l2_loss(W2) + tf.nn.l2_loss(b2)

    return l2_regularization, y_2

l2_regularization, y_ = multi_NN(feature, x, d)
y_ = tf.reshape(y_, [-1])

prediction = tf.greater(tf.sigmoid(y_), 0.5, name="prediction")
correct = tf.equal(prediction, tf.equal(y, True))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
if l2_regularizer_use == 1:
	loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(y, tf.float32),logits=y_) \
								+ l2_regularization
else: loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(y, tf.float32), logits=y_)
loss_mean = tf.reduce_mean(loss)
print loss_mean
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
    pos_x = x[np.argwhere(label==1)][:,0].tolist()
    pos_y = y[np.argwhere(label==1)][:,0].tolist()
    neg_x = x[np.argwhere(label==0)][:,0].tolist()
    neg_y = y[np.argwhere(label==0)][:,0].tolist()
    train_xx = np.array(pos_x[:int(len(pos_x)*t1)] + pos_x[int(len(pos_x)*(t2)):]\
                         + neg_x[:int(len(neg_x)*t1)] + neg_x[int(len(neg_x)*(t2)):])
    train_yy = np.array(pos_y[:int(len(pos_y)*t1)] + pos_y[int(len(pos_y)*(t2)):]\
                         + neg_y[:int(len(neg_y)*t1)] + neg_y[int(len(neg_y)*(t2)):])
    test_x = np.array(pos_x[int(len(pos_x)*t1):int(len(pos_x)*(t2))]\
                         + neg_x[int(len(neg_x)*t1):int(len(neg_x)*(t2))])
    test_y = np.array(pos_y[int(len(pos_y)*t1):int(len(pos_y)*(t2))]\
                         + neg_y[int(len(neg_y)*t1):int(len(neg_y)*(t2))])
    idx = np.arange(len(train_xx))
    np.random.shuffle(idx)
    train_x = train_xx[idx]
    train_y = train_yy[idx]
    test_idx = np.arange(len(test_y))
    np.random.shuffle(test_idx)
    test_x = test_x[test_idx]
    test_y = test_y[test_idx]
    print len(train_x)
    print len(test_x)
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
for k in range(int(1/test_ratio)):
	sess.run(init)
	train_x, train_y, test_x, test_y = cross_validation(data,label,k,test_ratio)
        #TRAIN
	for i in tqdm(range(itr), desc="training"):
                batch_x, batch_y = random_batch(train_x, train_y, N)
                _, acc, loss, train_sum =\
                            sess.run([train_step, accuracy, loss_mean, merged],\
			    feed_dict={x: batch_x, y: batch_y})
		if (i%1000 == 0 and i!=0) or i==-1:
			train_writer.add_summary(train_sum, i)
			print "[*] batch_time: " , i , "Accuracy: ", acc, "Loss: ", loss
	
        #TEST
	test_acc, test_loss, test_pred, test_label = \
                    sess.run([accuracy, loss_mean, prediction, y], \
	            feed_dict={x: test_x, y: test_y})
	fpr, tpr, thresholds = metrics.roc_curve(test_label, test_pred, pos_label=1)
        pdb.set_trace()
	test_auc = metrics.auc(fpr, tpr)
	print "[*] Fold", k ,"Test Accuracy: ", test_acc , ", loss: ", test_loss, "\n" 
	total_test_acc.append(test_acc)
	total_test_auc.append(test_auc)

print "[*]Average test accuracy", sum(total_test_acc)/len(total_test_acc)
print "[*]Average test auc", sum(total_test_auc)/len(total_test_auc)
