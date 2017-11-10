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
	print data_dir
	f.close()
	return data, feature, samples

rna_data, rna_feature, samples = file_read(data_dir_rna)
beta_data, beta_feature, _ = file_read(data_dir_beta)
label = np.zeros([100,3])
for i,sample in enumerate(samples):
    if 'msdp' in sample:
        label[i,2] = 1
    else:
        if 'sdpc' in sample:
            label[i,0] = 1
        else:
            label[i,1] = 1

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

msdp_x = data[np.argwhere(label[:,2]==1)][:,0,:].tolist()
msdp_y = label[np.argwhere(label[:,2]==1)][:,0,:].tolist()
sdp_x = data[np.argwhere(label[:,1]==1)][:,0,:].tolist()
sdp_y = label[np.argwhere(label[:,1]==1)][:,0,:].tolist()
sdpc_x = data[np.argwhere(label[:,0]==1)][:,0,:].tolist()
sdpc_y = label[np.argwhere(label[:,0]==1)][:,0,:].tolist()

print("# of samples:", len(samples))
print("Dimension of data:", data.shape)
print("# of MSDP:", len(np.nonzero(label[:,2])))
print("# of SDPC:", len(np.nonzero(label[:,0])))
print("# of SDP:", len(np.nonzero(label[:,1])))

# Model

N, itr, lr, train_log, d, test_ratio, l2_regularizer_use = config.getint('Parameter', 'batch_size'), config.getint('Parameter','iteration'), config.getfloat('Parameter', 'learning_rate'), config.get('Parameter', 'train_dir'), config.getint('Parameter', 'hidden_dim'), config.getfloat('Parameter', 'test_ratio'), config.getboolean('Parameter', 'l2_regularizer_use')
print "Batch Size: ", N
print "Learning rate: ", lr

g_step = tf.Variable(0)
lr = tf.train.exponential_decay(lr, g_step, 100, 0.98)
x = tf.placeholder(tf.float32, [None, len(feature)],name="x")
y = tf.placeholder(tf.int32, [None,3], name="y")

W = tf.Variable(tf.zeros([len(feature),3]), name='W1')
b = tf.Variable(tf.zeros(3), name="b")
l2_regularization = tf.nn.l2_loss(W) + tf.nn.l2_loss(b)
y_ = tf.nn.softmax(tf.matmul(x, W) + b) #[N,3]
prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1), name="prediction")
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

if l2_regularizer_use == 1:
	loss = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=tf.matmul(x, W) + b) + l2_regularization
else: loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=tf.matmul(x, W) + b)
loss_mean = tf.reduce_mean(loss)
train_step = tf.train.RMSPropOptimizer(lr, 0.9).minimize(loss_mean)

#Summary
tf.summary.scalar('loss', loss_mean)
tf.summary.scalar('accuracy', accuracy)
merged = tf.summary.merge_all()

saver = tf.train.Saver()
init = tf.global_variables_initializer()
sess = tf.Session()
train_writer = tf.summary.FileWriter(train_log, sess.graph)
sess.run(init)

def cross_validation(msdp_x,msdp_y,sdp_x,sdp_y,sdpc_x,sdpc_y,portion,test_ratio):
    t1 = test_ratio*portion
    t2 = test_ratio*(portion+1)
    train_xx = np.array(msdp_x[:int(len(msdp_x)*t1)] + msdp_x[int(len(msdp_x)*(t2)):] \
                        + sdp_x[:int(len(sdp_x)*t1)] + sdp_x[int(len(sdp_x)*(t2)):]\
                        + sdpc_x[:int(len(sdpc_x)*t1)] + sdpc_x[int(len(sdpc_x)*(t2)):])
    
    train_yy = np.array(msdp_y[:int(len(msdp_y)*t1)] + msdp_y[int(len(msdp_y)*(t2)):] \
                        + sdp_y[:int(len(sdp_y)*t1)] + sdp_y[int(len(sdp_y)*(t2)):]\
                        + sdpc_y[:int(len(sdpc_y)*t1)] + sdpc_y[int(len(sdpc_y)*(t2)):])
    
    test_x =  np.array(msdp_x[int(len(msdp_x)*t1):int(len(msdp_x)*(t2))] \
                        + sdp_x[int(len(sdp_x)*t1):int(len(sdp_x)*(t2))]\
                        + sdpc_x[int(len(sdpc_x)*t1):int(len(sdpc_x)*(t2))])
    test_y =  np.array(msdp_y[int(len(msdp_y)*t1):int(len(msdp_y)*(t2))] \
                        + sdp_y[int(len(sdp_y)*t1):int(len(sdp_y)*(t2))]\
                        + sdpc_y[int(len(sdpc_y)*t1):int(len(sdpc_y)*(t2))])
    idx = np.arange(len(train_xx))
    np.random.shuffle(idx)
    train_x = train_xx[idx]
    train_y = train_yy[idx]
    print len(train_x), len(test_x)

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
	train_x, train_y, test_x, test_y = \
                        cross_validation(msdp_x,msdp_y,sdp_x,sdp_y,sdpc_x,sdpc_y,k,test_ratio)
	#TRAIN
	for i in range(itr):
		batch_x, batch_y = random_batch(train_x, train_y, N)
		_, acc, loss, train_sum, weight = sess.run([train_step, accuracy, loss_mean, merged, W],\
															feed_dict={x: batch_x, y: batch_y})
		if i%100 == 0 and i!=0:
                    print "iteration: ",i," acc: ", acc, " loss: ", loss
                #	train_writer.add_summary(train_sum, i)
		#	print "[*] batch_time: " , i , "Accuracy: ", acc, "Loss: ", loss
		#	max_idx = np.argsort(weight,axis=0)
                #       for j in range(5):
                #            print j, "rank \n",
                #            print "Min: ", float(weight[int(max_idx[j])]), \
                #                            '\t', feature[int(max_idx[j])]
                #            print "Max: ", float(weight[int(max_idx[-(j+1)])]), \
                #                            '\t', feature[int(max_idx[-(j+1)])]
	
        #TEST
        test_acc, test_loss, test_pred, label = \
                sess.run([accuracy, loss_mean, prediction, y], feed_dict={x:test_x, y:test_y})
	#fpr, tpr, thresholds = metrics.roc_curve(label, test_pred, pos_label=1)
	#test_auc = metrics.auc(fpr, tpr)
	print "[*] Fold", k ,"Test Accuracy: ", test_acc , ", loss: ", test_loss, "\n" 
	total_test_acc.append(test_acc)
	#total_test_auc.append(test_auc)
        with open(save_result_dir, 'a') as f:
            f.write(str(test_acc))
            f.write('\n')

print "[*]Average test accuracy", sum(total_test_acc)/len(total_test_acc)
#print "[*]Average test auc", sum(total_test_auc)/len(total_test_auc)
with open(save_result_dir, 'a') as f:
    f.write(str(sum(total_test_acc)/len(total_test_acc)))
    f.write('\n')
