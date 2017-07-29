from __future__ import absolute_import
from __future__ import division
from __future__	import print_function

import numpy as np
import argparse, os
import tensorflow as tf

from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.contrib.rnn import BasicLSTMCell, BasicRNNCell, GRUCell
from EURNN import EURNNCell
from GORU import GORUCell
from rotational_models import GRRUCell, BasicLSTRMCell
from supercell_DRUM import HyperLSTMCell, HyperDRUMCell
from drum import DRUMCell

from ptb_iterator import *
import re
import sys

#read data
def file_data(stage, n_batch, n_data, T, n_epochs, vocab_to_idx = None):
	if stage == 'train':
		file_name = 'data/ptb.train.txt'
	elif stage == 'valid':
		file_name = 'data/ptb.valid.txt'	
	elif stage == 'test':
		file_name = 'data/ptb.test.txt'

	with open(file_name, 'r') as f:
		raw_data = f.read()
	raw_data = raw_data.replace('\n', '')
	print("Data length: " , len(raw_data))

	if vocab_to_idx == None:
		vocab = set(raw_data)
		print(vocab)
		vocab_size = len(vocab)
		print("Vocab size: ", vocab_size)

		my_dict = {}
		idx_to_vocab= {}
		vocab_to_idx = {}
		for index, item in enumerate(vocab):
			idx_to_vocab[index] = item
			vocab_to_idx[item] = index

	data = [vocab_to_idx[c] for c in raw_data][:n_data]
	print("Total data length: " , len(data))

	if T == -1: 
		T = len(data) - 1

	def gen_epochs(n, numsteps, n_batch):
		for i in range(n):
			yield ptb_iterator(data, n_batch, numsteps)
	
	print("Sequence length: ", T)
	myepochs = gen_epochs(n_epochs, T, n_batch)

	return myepochs, vocab_to_idx

def main(model, T, n_epochs, n_batch, n_hidden, learning_rate, decay, model_save_to)
	#create data 
	max_len_data = 1000000000
	epoch_train, vocab_to_idx = file_data('train', n_batch, max_len_data, T, n_epochs)
	n_input = len(vocab_to_idx)
	epoch_val, _ = file_data('valid', 1, max_len_data, -1, 10000, vocab_to_idx = vocab_to_idx)
	epoch_test, _ = file_data('test', 1, max_len_data, -1, 1, vocab_to_idx = vocab_to_idx)
	n_output = n_input

	# --- Create graph and compute gradients ----------------------
	x = tf.placeholder("int32", [None, None])
	y = tf.placeholder("int64", [None, None])
	
	input_data = tf.one_hot(x, n_input, dtype=tf.float32)
	print(input_data)
	input()

	#cell definition
	h = None
	if model == "LSTM":
		cell = BasicLSTMCell(n_hidden, state_is_tuple = True, forget_bias = 0.0, 
					         initial_state = h)
	if model == "DRUM": 
		cell = 	DRUMCell(n_hidden, size_batch = n_batch)
	hidden_out, states = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32, 
										   initial_state = h)

	#hidden to output
	V_init_val = 0.1
	V_weights = tf.get_variable("V_weights", shape = [n_hidden, n_output], \
			dtype=tf.float32, initializer = tf.random_uniform_initializer(-V_init_val, V_init_val))
	V_bias = tf.get_variable("V_bias", shape = [n_output], \
			dtype=tf.float32, initializer = tf.constant_initializer(0.1))
	hidden_out_list = tf.unstack(hidden_out, axis = 1)
	temp_out = tf.stack([tf.matmul(i, V_weights) for i in hidden_out_list])
	output_data = tf.nn.bias_add(tf.transpose(temp_out, [1,0,2]), V_bias) 	

	#evaluation of the process
	cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_data, labels=y))
	correct_pred = tf.equal(tf.argmax(output_data, 2), y)
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


	# --- Initialization ----------------------
	if learning_rate_decay:
		global_step = tf.Variable(0, trainable = False)
		starter_learning_rate = learning_rate
		learning_rate_final = tf.train.exponential_decay(starter_learning_rate, global_step,
    		                                       10000, 0.7, staircase = True)
	elif piecewise_rate_decay:
		global_step = tf.Variable(0, trainable = False)
		boundaries = [3000, 70000]
		values = [0.01, 0.001, 0.0001]
		learning_rate_final = tf.train.piecewise_constant(global_step, boundaries, values)
	else:
		learning_rate_final = learning_rate

	if optimization == "RMSProp":
		optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate_final, decay=decay).minimize(cost)
	elif optimization == "Momentum": 
		optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate_final, momentum=decay).minimize(cost)
	elif optimization == "Adam": 
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_final).minimize(cost)
	init = tf.global_variables_initializer()
	
	print("\n###")
	sumz = 0 
	for i in tf.global_variables():
		print(i.name)
		print(i.name, i.shape, np.prod(np.array(i.get_shape().as_list())))
		sumz += np.prod(np.array(i.get_shape().as_list()))
	print("# parameters: ", sumz)
	print("###\n")
	## input() 


	# --- save result ----------------------
	filename = "./output/character/T=" + str(T) + "/" + model  + "_N=" + str(n_hidden) + "_lambda=" + str(learning_rate) + "_E=" + str(n_epochs) #"_beta=" + str(decay)
	if istanh: 
		print("Tanh!")
		filename += "_tanh"
	if isdropout: 
		print("Dropout!")
		filename += "_dropout"
	if learning_rate_decay: 
		print("Learning rate decay!")
		filename += "_lrd"
	if piecewise_rate_decay: 
		print("Piecewise rate decay!")
		filename += "_prd"
	
	if model == "EURNN"  or model == "GORU":
		print(model)
		if FFT:
			filename += "_FFT"
		else:
			filename = filename + "_L=" + str(capacity)
	if model == "GRRU": 
		print(model)
		if ismodrelu: 
			filename += "_modrelu_" 
			filename += str(modrelu_const)

	filename = filename + ".txt"
	if not os.path.exists(os.path.dirname(filename)):
		try:
			os.makedirs(os.path.dirname(filename))
		except OSError as exc: # Guard against race condition
			if exc.errno != errno.EEXIST:
				raise
	f = open(filename, 'w')
	f.write("########\n\n")
	f.write("## \tModel: %s with N=%d"%(model, n_hidden))
	if model == "EURNN" or model == "GORU":
		if FFT:
			f.write(" FFT")
		else:
			f.write(" L=%d"%(capacity))
	f.write("\n\n")
	f.write("########\n\n")

	# --- Training Loop ---------------------------------------------------------------
	if model_save_to == "my-model":
		print("Autogenerating the save name")
		model_save_to = "nlp_"+str(model)+"_"+str(n_hidden)+str(istanh)
		print("Save name is: " , model_save_to)
		savename="./output/nlp/"+str(model_save_to)

		if not os.path.exists(os.path.dirname(savename)):
			try:
				os.makedirs(os.path.dirname(savename))
			except OSError as exc: # Guard against race condition
				if exc.errno != errno.EEXIST:
					raise

	def do_validation(m1, m2):
		j = 0
		val_losses = []
		val_accuracies = [] 
		for val in epoch_val:
			j +=1 
			if j >= 2:
				break
			print("Running validation...")
			val_state = None
			for stepb, (X_val,Y_val) in enumerate(val):
				val_batch_x = X_val
				val_batch_y = Y_val
				val_dict = {x:val_batch_x,y:val_batch_y}
				#if val_state is not None:
					#This needs to be initialized from the original net creation. 
				#	val_dict[h] = val_state
				if notstates:
					val_acc,val_loss = sess.run([accuracy,cost],feed_dict=val_dict)
				else:
					val_acc, val_loss, val_state = sess.run([accuracy, cost,states],feed_dict=val_dict)
				val_losses.append(val_loss)
				val_accuracies.append(val_acc)
		print("Validations:", )
		
		#averaging process here...
		validation_losses.append(sum(val_losses)/len(val_losses))
		validation_accuracies.append(sum(val_accuracies)/len(val_accuracies))
		print("Validation Loss= " + \
					  "{:.6f}".format(validation_losses[-1]) + 
					  ", Validation Accuracy= " + \
				  	  "{:.5f}".format(validation_accuracies[-1]))

		f.write("%d\t%f\t%f\t%f\t%f\n"%(t, validation_losses[-1], 
				validation_accuracies[-1], m1, m2))
		f.flush()

	saver = tf.train.Saver()

	step = 0
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = memory)
	with tf.Session(config = tf.ConfigProto(log_device_placement = False, 
										    allow_soft_placement = False,
										    gpu_options = gpu_options)) as sess:
		print("Session Created")
		# if loadFrom != "":
		# 	new_saver = tf.train.import_meta_graph(loadFrom+'.meta')
		# 	new_saver.restore(sess, tf.train.latest_checkpoint('./'))
		# 	print("Session loaded from: " , loadFrom)
		# else:
		# 	#summary_writer = tf.train.SummaryWriter('/tmp/logdir', sess.graph)
		# 	sess.run(init)
		

		steps = []
		losses = []
		accs = []
		validation_losses = []
		validation_accuracies = [] 

		sess.run(init)
		training_state = None
		i = 0
		t = 0
		mx2 = 0
		for epoch in epoch_train:
			print("Epoch: " , i)
			for step, (X,Y) in enumerate(epoch):
				batch_x = X
				batch_y = Y
				myfeed_dict={x: batch_x, y: batch_y}

				if training_state is not None:
					myfeed_dict[h] = training_state

				if notstates:
					_, acc, loss = sess.run([optimizer, accuracy, cost], feed_dict = myfeed_dict)
				else:
					empty, acc, loss, training_state = sess.run([optimizer, accuracy, cost, states], feed_dict = myfeed_dict)

				print("Iter " + str(t) + ", Minibatch Loss= " + \
					  "{:.6f}".format(loss) + ", Training Accuracy= " + \
				  	  "{:.5f}".format(acc))
				
				steps.append(t)
				losses.append(loss)
				accs.append(acc)
				t += 1

				## output hidden value
				tmp = sess.run(hidden_out, feed_dict={x: batch_x, y: batch_y})
				print(tmp.size, tmp.shape)
				mx = 0
				for i in range(T):
					if np.abs(np.average(tmp[0][i])) > mx: 
						mx = np.abs(np.average(tmp[0][i]))
				print("Max for iteration: ", mx)
				if mx > mx2: 
					mx2 = mx
				print("Max for whole: ", mx2)
				## 

				if np.isnan(loss) or loss > 30000.0: 
					f.write("Encountered a NaN blow up! Fix the model/parameters...\n")
					f.write("The maximal norms: " + str(mx2) + " " + str(mx) + "\n")
					print("Sorry, a blow up!")
					sys.exit()

				#finer validation!!
				if step % 10 == 9:
					do_validation(mx, mx2)
					saver.save(sess,savename)
			i += 1
		print("Optimization Finished!")
		
		


		j = 0
		test_losses = []
		test_accuracies = []
		for test in epoch_test:
			j +=1 
			if j >= 2:
				break
			print("Running validation...")
			test_state = None
			for stepb, (X_test,Y_test) in enumerate(test):
				test_batch_x = X_test
				test_batch_y = Y_test
				test_dict = {x:test_batch_x,y:test_batch_y}
				# if test_state is not None:
					#This needs to be initialized from the original net creation. 
					# test_dict[h] = test_state
				test_acc, test_loss = sess.run([accuracy, cost],feed_dict=test_dict)
				test_losses.append(test_loss)
				test_accuracies.append(test_acc)
		print("test:", )
		test_losses.append(sum(test_losses)/len(test_losses))
		test_accuracies.append(sum(test_accuracies)/len(test_accuracies))
		print("test Loss= " + \
				  "{:.6f}".format(test_losses[-1]))
		f.write("Test result: %d\t%f\t%f\n"%(t, test_losses[-1], test_accuracies[-1]))
		f.write("The maximal norms: " + str(mx2) + " " + str(mx) + "\n")
"""
