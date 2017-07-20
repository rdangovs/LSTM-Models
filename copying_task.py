from __future__ import absolute_import
from __future__ import division
from __future__	import print_function

import numpy as np
import argparse, os
import tensorflow as tf
import sys

from tensorflow.contrib.rnn import BasicLSTMCell, BasicRNNCell, GRUCell
from EURNN import EURNNCell
from GORU import GORUCell
from LSTSM import BasicLSTSMCell
from LSTUM import BasicLSTUMCell
from rotational_models import GRRUCell, BasicLSTRMCell
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops

sigmoid = math_ops.sigmoid 
tanh = math_ops.tanh
matm = math_ops.matmul
mul = math_ops.multiply 
relu = nn_ops.relu

def random_variable(shape, dev): 
  initial = tf.truncated_normal(shape, stddev= dev)
  return tf.Variable(initial)

def copying_data(T, n_data, n_sequence):
	seq = np.random.randint(1, high=9, size=(n_data, n_sequence))
	zeros1 = np.zeros((n_data, T-1))
	zeros2 = np.zeros((n_data, T))
	marker = 9 * np.ones((n_data, 1))
	zeros3 = np.zeros((n_data, n_sequence))

	x = np.concatenate((seq, zeros1, marker, zeros3), axis=1).astype('int32')
	y = np.concatenate((zeros3, zeros2, seq), axis=1).astype('int64')
	
	return x, y

def main(model, T, n_iter, n_batch, n_hidden, capacity, comp, FFT, learning_rate, decay,  
	     ismodrelu, modrelu_const, istanh, learning_rate_decay):
	learning_rate = float(learning_rate)
	
	## makes the modReLU negative
	modrelu_const *= -1 
	#

	decay = float(decay)

	# --- Set data params ----------------
	n_input = 10
	n_output = 9
	n_sequence = 10
	n_train = n_iter * n_batch
	n_test = n_batch

	n_steps = T+20
	n_classes = 9


  	# --- Create data --------------------

	train_x, train_y = copying_data(T, n_train, n_sequence)
	test_x, test_y = copying_data(T, n_test, n_sequence)


	# --- Create graph and compute gradients ----------------------
	x = tf.placeholder("int32", [None, n_steps])
	y = tf.placeholder("int64", [None, n_steps])
	
	input_data = tf.one_hot(x, n_input, dtype=tf.float32)



	# --- Input to hidden layer ----------------------
	if model == "LSTM":
		cell = BasicLSTMCell(n_hidden, state_is_tuple=True, forget_bias=1)
		hidden_out, _ = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
	elif model == "LSTSM":
		cell = BasicLSTSMCell(n_hidden, forget_bias=1)
		hidden_out, _ = tf.nn.dynamic_rnn(cell, input_data, 
										  initial_state=
										  (LSTMStateTuple
										  	(random_variable([n_batch, n_hidden], 0.1), 
										  	 random_variable([n_batch, n_hidden], 0.1))), 
										  dtype=tf.float32)
	elif model == "LSTUM":
		cell = BasicLSTUMCell(n_hidden, size_batch = n_batch, forget_bias=1)
		hidden_out, _ = tf.nn.dynamic_rnn(cell, input_data, 
										  initial_state=
										  (LSTMStateTuple
										  	(random_variable([n_batch,n_hidden ** 2], 0.1), 
										  	 random_variable([n_batch, n_hidden], 0.1))), 
										  dtype=tf.float32)
	elif model == "LSTRM":
		if ismatrix: 
			cell = BasicLSTRMCell(n_hidden, size_batch = n_batch, forget_bias=1, isMatrix=True)
			hidden_out, _ = tf.nn.dynamic_rnn(cell, input_data, 
											  initial_state =
											  (LSTMStateTuple
											  	(random_variable([n_batch, n_hidden ** 2], 0.1), 
											  	 random_variable([n_batch, n_hidden], 0.1))), 
											  dtype = tf.float32)
		else: 
			cell = BasicLSTRMCell(n_hidden, size_batch = n_batch, forget_bias=1, isMatrix=False)
			hidden_out, _ = tf.nn.dynamic_rnn(cell, input_data, 
										  	initial_state =
										  	(LSTMStateTuple
										  		(random_variable([n_batch, n_hidden], 0.1), 
										  	 	random_variable([n_batch, n_hidden], 0.1))), 
										  	dtype = tf.float32)
	elif model == "GRU":
		cell = GRUCell(n_hidden)
		hidden_out, _ = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
	elif model == "GRRU":
		cell = GRRUCell(n_hidden, size_batch = n_batch, is_modrelu = ismodrelu, modrelu_const = modrelu_const)
		hidden_out, _ = tf.nn.dynamic_rnn(cell, input_data, dtype = tf.float32)
	elif model == "RNN":
		cell = BasicRNNCell(n_hidden)
		hidden_out, _ = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
	elif model == "EURNN":
		cell = EURNNCell(n_hidden, capacity, FFT, comp)
		if comp:
			hidden_out_comp, _ = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.complex64)
			hidden_out = tf.real(hidden_out_comp)
		else:
			hidden_out, _ = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
	elif model == "GORU":
		cell = GORUCell(n_hidden, capacity, FFT)
		hidden_out, _ = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)

	# --- Hidden Layer to Output ----------------------
	# important `tanh` prevention from blow up 
	if istanh: 
		hidden_out = tanh(hidden_out)

	V_init_val = np.sqrt(6.)/np.sqrt(n_output + n_input)

	V_weights = tf.get_variable("V_weights", shape = [n_hidden, n_classes], dtype=tf.float32, initializer=tf.random_uniform_initializer(-V_init_val, V_init_val))
	V_bias = tf.get_variable("V_bias", shape=[n_classes], dtype=tf.float32, initializer=tf.constant_initializer(0.01))

	hidden_out_list = tf.unstack(hidden_out, axis=1)
	temp_out = tf.stack([tf.matmul(i, V_weights) for i in hidden_out_list])
	output_data = tf.nn.bias_add(tf.transpose(temp_out, [1,0,2]), V_bias) 

	# --- evaluate process ----------------------
	cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_data, labels=y))
	correct_pred = tf.equal(tf.argmax(output_data, 2), y)
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


	# --- Initialization ----------------------
	##
	if learning_rate_decay:
		global_step = tf.Variable(0, trainable = False)
		starter_learning_rate = learning_rate
		learning_rate_final = tf.train.exponential_decay(starter_learning_rate, global_step,
    		                                       10000, 0.7, staircase = True)
	else: 
		learning_rate_final = learning_rate
	##

	optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=decay).minimize(cost)
	init = tf.global_variables_initializer()

	print("\n###")
	sumz = 0 
	for i in tf.global_variables():
		#print(i.name)
		print(i.name, i.shape, np.prod(np.array(i.get_shape().as_list())))
		sumz += np.prod(np.array(i.get_shape().as_list()))
	print("# parameters: ", sumz)
	print("###\n")
	#input()

	# --- save result ----------------------
	filename = "./output/copying/T=" + str(T) + "/" + model  + "_N=" + str(n_hidden) + "_lambda=" + str(learning_rate) + "_decay=" + str(decay)
	
	if istanh: 
		print("Tanh!")
		filename += "_tanh"
	if learning_rate_decay: 
		print("Learning rate decay!")
		filename += "_lrd"
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


	# --- Training Loop ----------------------
	mx2  = 0
	step = 0
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.10)
	with tf.Session(config = tf.ConfigProto(log_device_placement = False, 
										    allow_soft_placement = False,
										    gpu_options = gpu_options)) as sess:
		sess.run(init)

		steps = []
		losses = []
		accs = []


		while step < n_iter:
			batch_x = train_x[step * n_batch : (step+1) * n_batch]
			batch_y = train_y[step * n_batch : (step+1) * n_batch]


			sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

			acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
			loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})

			if np.isnan(loss) or loss > 30000.0: 
					f.write("Encountered a NaN blow up! Fix the model/parameters...\n")
					print("Sorry, a blow up!")
					sys.exit()

			print("Iter " + str(step) + ", Minibatch Loss= " + \
				  "{:.6f}".format(loss) + ", Training Accuracy= " + \
				  "{:.5f}".format(acc))


			steps.append(step)
			losses.append(loss)
			accs.append(acc)
			step += 1

			f.write("%d\t%f\t%f\n"%(step, loss, acc))


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


		print("Optimization Finished!")


		
		# --- test ----------------------

		test_acc = sess.run(accuracy, feed_dict={x: test_x, y: test_y})
		test_loss = sess.run(cost, feed_dict={x: test_x, y: test_y})
		f.write("Test result: Loss= " + "{:.6f}".format(test_loss) + \
					", Accuracy= " + "{:.5f}".format(test_acc))


if __name__=="__main__":
	parser = argparse.ArgumentParser(
		description="Copying Task")
	parser.add_argument("model", default='LSTM', help='Model name: LSTM, LSTSM, LSTRM, LSTUM, EURNN, GRU, GRRU, GORU, GRRU')
	parser.add_argument('-T', type=int, default=200, help='Information sequence length')
	parser.add_argument('--n_iter', '-I', type=int, default=20000, help='training iteration number')
	parser.add_argument('--n_batch', '-B', type=int, default=128, help='batch size')
	parser.add_argument('--n_hidden', '-H', type=int, default=100, help='hidden layer size')
	parser.add_argument('--capacity', '-L', type=int, default=2, help='Tunable style capacity, only for EURNN, default value is 2')
	parser.add_argument('--comp', '-C', type=str, default="False", help='Complex domain or Real domain. Default is False: real domain')
	parser.add_argument('--FFT', '-F', type=str, default="False", help='FFT style, default is False')
	parser.add_argument('--learning_rate', '-R', default=0.001, type=str)
	parser.add_argument('--decay', '-D', default=0.9, type=str)
	parser.add_argument('--ismodrelu', '-Mo', default="False", type=str)
	parser.add_argument('--modrelu_const', '-Co', default=0.0, type=float)	
	parser.add_argument('--istanh', '-Ta', default="False", type=str)	
	parser.add_argument('--learning_rate_decay', '-RD', default="False", type=str)	
	
	args = parser.parse_args()
	dict = vars(args)

	for i in dict:
		if (dict[i]=="False"):
			dict[i] = False
		elif dict[i]=="True":
			dict[i] = True
		
	kwargs = {	
				'model': dict['model'],
				'T': dict['T'],
				'n_iter': dict['n_iter'],
			  	'n_batch': dict['n_batch'],
			  	'n_hidden': dict['n_hidden'],
			  	'capacity': dict['capacity'],
			  	'comp': dict['comp'],
			  	'FFT': dict['FFT'],
			  	'learning_rate': dict['learning_rate'],
			  	'decay': dict['decay'],
			  	'ismodrelu': dict['ismodrelu'],
			  	'modrelu_const': dict['modrelu_const'],
			  	'istanh': dict['istanh'],
			  	'learning_rate_decay': dict['learning_rate_decay']
			}

	main(**kwargs)
