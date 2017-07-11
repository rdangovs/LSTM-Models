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

from ptb_iterator import *
import re
import sys

word_level = False
notstates = False

sigmoid = math_ops.sigmoid 
tanh = math_ops.tanh
matm = math_ops.matmul
mul = math_ops.multiply 
relu = nn_ops.relu
sign = math_ops.sign

def file_data(stage, n_batch, n_data, T, n_epochs, vocab_to_idx,readIntegers=True):

	if not word_level:
		if stage == 'train':
			file_name = 'data/ptb.train.txt'
		elif stage == 'valid':
			file_name = 'data/ptb.valid.txt'	
		elif stage == 'test':
			file_name = 'data/ptb.test.txt'
	else:
		if stage == 'train':
			file_name = 'data/ptb.train_words_nodigits_new.txt'
		elif stage == 'valid':
			file_name = 'data/ptb.valid_words_nodigits_new.txt'	
		elif stage == 'test':
			file_name = 'data/ptb.test_words_nodigits_new.txt'

	with open(file_name,'r') as f:
		raw_data = f.read()
		print("Data length: " , len(raw_data))

	if word_level:
		if readIntegers:
			raw_data = [int(line.rstrip('\n')) for line in open(file_name)]


	if vocab_to_idx == None:
		vocab = set(raw_data)
		vocab_size = len(vocab)
		print("Vocab size: ", vocab_size)

		my_dict = {}
		idx_to_vocab= {}
		vocab_to_idx = {}
		for index, item in enumerate(vocab):
			idx_to_vocab[index] = item
			vocab_to_idx[item] = index
	#idx_to_vocab = dict(enumerate(vocab))
	#now flip this dictionary
	#Flip this dictionary
	#vocab_to_idx = dict(zip(idx_to_vocab.values(), idx_to_vocab.keys()))
	data = [vocab_to_idx[c] for c in raw_data][:n_data]
	print("Total data length: " , len(data))

	# print(data[0:1000])

	#Deal with batch size >.<

	#Numsteps is your sequence length. In this case the earlier formula. 
	def gen_epochs(n, numsteps, n_batch):
		for i in range(n):
			yield ptb_iterator(data, n_batch, numsteps)
	
	print("Sequence length: ", T)
	myepochs = gen_epochs(n_epochs, T, n_batch)
	print(myepochs)

	return myepochs, vocab_to_idx

# file_data('train', 20, 10000000, 50, 20)

def main(model, T, n_epochs, n_batch, n_hidden, capacity, comp, FFT, learning_rate, decay, ismodrelu, modrelu_const, istanh, islin):
	# --- Set data params ----------------
	
	## makes the modReLU negative
	modrelu_const *= -1 
	#


	#Create Data
	max_len_data = 1000000000
	epoch_train, vocab_to_idx = file_data('train', n_batch, max_len_data, T, n_epochs, None)
	n_input = len(vocab_to_idx)
	epoch_val, _ = file_data('valid', n_batch, max_len_data, T, 10000, vocab_to_idx)
	epoch_test, _ = file_data('test', n_batch, max_len_data, T, 1, vocab_to_idx)
	n_output = n_input


	# --- Create graph and compute gradients ----------------------
	x = tf.placeholder("int32", [None, T])
	y = tf.placeholder("int64", [None, T])

	n_embed = 30
	if (word_level):
		embed_init_val = np.sqrt(6.)/np.sqrt(n_input)
		embed = tf.get_variable('Embedding', [n_input, n_embed] ,initializer = init_ops.random_normal_initializer(-embed_init_val, embed_init_val), dtype=tf.float32)
		input_data = tf.nn.embedding_lookup(embed, x)
		n_input = n_embed

	else:	
		input_data = tf.one_hot(x, n_input, dtype=tf.float32)


	# Input to hidden layer
	cell = None
	h = None
	#h_b = None
	if model == "LSTM":
		cell = BasicLSTMCell(n_hidden, state_is_tuple=True, forget_bias=1)
		if h == None:
			h = cell.zero_state(n_batch,tf.float32)
		hidden_out, states = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
	elif model == "GRU":
		cell = GRUCell(n_hidden)
		if h == None:
			h = cell.zero_state(n_batch,tf.float32)
		hidden_out, states = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
	elif model == "GRRU":
		cell = GRRUCell(n_hidden, size_batch = n_batch, is_modrelu = ismodrelu, modrelu_const = modrelu_const, is_lin = islin)
		if h == None:
			h = cell.zero_state(n_batch,tf.float32)
		hidden_out, states = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
	elif model == "RNN":
		cell = BasicRNNCell(n_hidden)
		if h == None:
			h = cell.zero_state(n_batch,tf.float32)
		hidden_out, states = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
	elif model == "EURNN":
		cell = EURNNCell(n_hidden, capacity, FFT, comp)
		if h == None:
			h = cell.zero_state(n_batch,tf.float32)
		if comp:
			hidden_out_comp, states = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.complex64)
			hidden_out = tf.real(hidden_out_comp)
		else:
			hidden_out, states = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
	elif model == "GORU":
		cell = GORUCell(n_hidden, capacity, FFT, comp)
		if h == None:
			h = cell.zero_state(n_batch,tf.float32)
		if comp:
			hidden_out_comp, states = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.complex64)
			hidden_out = tf.real(hidden_out_comp)
		else:
			hidden_out, states = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
	


	# Hidden Layer to Output
	V_init_val = np.sqrt(6.)/np.sqrt(n_output + n_input)

	V_weights = tf.get_variable("V_weights", shape = [n_hidden, n_output], \
			dtype=tf.float32, initializer=tf.random_uniform_initializer(-V_init_val, V_init_val))
	V_bias = tf.get_variable("V_bias", shape=[n_output], \
			dtype=tf.float32, initializer=tf.constant_initializer(0.01))

	if istanh: 
		hidden_out = tanh(hidden_out)
	hidden_out_list = tf.unstack(hidden_out, axis=1)
	temp_out = tf.stack([tf.matmul(i, V_weights) for i in hidden_out_list])
	output_data = tf.nn.bias_add(tf.transpose(temp_out, [1,0,2]), V_bias) 


	# define evaluate process
	cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_data, labels=y))
	correct_pred = tf.equal(tf.argmax(output_data, 2), y)
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


	# --- Initialization ----------------------
	optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=decay).minimize(cost)
	init = tf.global_variables_initializer()

	print("\n###")
	sum = 0 
	for i in tf.global_variables():
		#print(i.name)
		print(i.name, i.shape, np.prod(np.array(i.get_shape().as_list())))
		sum += np.prod(np.array(i.get_shape().as_list()))
	print("# parameters: ", sum)
	print("###\n")
	input()

	# --- save result ----------------------
	filename = "./output/character/T=" + str(T) + "/" + model  + "_N=" + str(n_hidden) + "_lambda=" + str(learning_rate) + "_beta=" + str(decay)
	
	if istanh: 
		print("Tanh!")
		filename += "_tanh"
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
		if islin: 
			filename += "_lin"

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


	

	# --- baseline -----
	
	# --- Training Loop ---------------------------------------------------------------


	# if saveTo == "my-model":
	# 	print("Autogenerating the save name")
	# 	saveTo = "nlp_"+str(model)+"_"+str(n_hidden)+"_"+str(capacity)+"_"+str(approx)+"_"+str(num_layers)
	# 	print("Save name is: " , saveTo)
	# 	savename="./output/nlp/"+str(saveTo)

	# 	if not os.path.exists(os.path.dirname(savename)):
	# 		try:
	# 			os.makedirs(os.path.dirname(savename))
	# 		except OSError as exc: # Guard against race condition
	# 			if exc.errno != errno.EEXIST:
	# 				raise



	def do_validation(m1, m2):
		j = 0
		val_losses = []
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
				if val_state is not None:
					#This needs to be initialized from the original net creation. 
					val_dict[h] = val_state
				if notstates:
					val_acc,val_loss = sess.run([accuracy,cost],feed_dict=val_dict)
				else:
					val_acc, val_loss, val_state = sess.run([accuracy, cost,states],feed_dict=val_dict)
				val_losses.append(val_loss)
		print("Validations:", )
		validation_losses.append(sum(val_losses)/len(val_losses))
		print("Validation Loss= " + \
				  "{:.6f}".format(validation_losses[-1]))
		
		f.write("%d\t%f\t%f\t%f\n"%(t, validation_losses[-1], m1, m2))
		f.flush()

	# saver = tf.train.Saver()

	step = 0
	with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=False)) as sess:
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

				# if training_state is not None:
				# #	#This needs to be initialized from the original net creation. 
					
				#myfeed_dict[h] = training_state
				# 	#print("State: " , training_state)
					#print("Comp : ", training_state[0])

					#print("Sum: " , sum([i*i for i in training_state[0]]))
				#print("Feed dict: " , myfeed_dict)
				if notstates:
					_, acc, loss = sess.run([optimizer, accuracy, cost], feed_dict = myfeed_dict)
				else:
					empty,acc,loss,training_state = sess.run([optimizer, accuracy, cost, states], feed_dict = myfeed_dict)
				#print("Sum: " , sum([i*i for i in training_state[0]]))

				print("Iter " + str(t) + ", Minibatch Loss= " + \
					  "{:.6f}".format(loss) + ", Training Accuracy= " + \
				  	  "{:.5f}".format(acc))
				
				steps.append(t)
				losses.append(loss)
				accs.append(acc)
				t += 1

				if np.isnan(loss) or loss > 30000.0: 
					f.write("Encountered a NaN blow up! Fix the model/parameters...\n")
					f.write("The maximal norms: " + str(mx2) + " " + str(mx) + "\n")
					print("Sorry, a blow up!")
					sys.exit()

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

				if step % 1000 == 999:
					do_validation(mx, mx2)
					# saver.save(sess,savename)
					#Now I need to take an epoch and go through it. I will average the losses at the end
						# f2.write("%d\t%f\t%f\n"%(step, loss, acc))
					# f.flush()
					# f2.flush()
				# mystates = sess.run(states, feed_dict=myfeed_dict)
				# print ("States",training_state)
			i += 1

		print("Optimization Finished!")
		
		


		j = 0
		test_losses = []
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
		print("test:", )
		test_losses.append(sum(test_losses)/len(test_losses))
		print("test Loss= " + \
				  "{:.6f}".format(test_losses[-1]))
		f.write("Test result: %d\t%f\n"%(t, test_losses[-1]))
		f.write("The maximal norms: " + str(mx2) + " " + str(mx) + "\n")

		


if __name__=="__main__":
	parser = argparse.ArgumentParser(
		description="Copying Memory Problem")

	parser.add_argument("model", default='LSTM', help='Model name: LSTM, EURNN, GRU, GORU')
	parser.add_argument('-T', type=int, default=50, help='T-gram')
	parser.add_argument("--n_epochs", '-E', type=int, default=20, help='num epochs')
	parser.add_argument('--n_batch', '-B', type=int, default=32, help='batch size')
	parser.add_argument('--n_hidden', '-H', type=int, default=512, help='hidden layer size')
	parser.add_argument('--capacity', '-L', type=int, default=2, help='Tunable style capacity, only for EURNN, default value is 2')
	parser.add_argument('--comp', '-C', type=str, default="False", help='Complex domain or Real domain. Default is False: real domain')
	parser.add_argument('--FFT', '-F', type=str, default="False", help='FFT style, default is False')
	parser.add_argument('--learning_rate', '-R', default=0.001, type=float)
	parser.add_argument('--decay', '-D', default=0.9, type=float)
	parser.add_argument('--ismodrelu', '-Mo', default="False", type=str)
	parser.add_argument('--modrelu_const', '-Co', default=0.0, type=float)	
	parser.add_argument('--istanh', '-Ta', default="False", type=str)	
	parser.add_argument('--islin', '-Li', default="False", type=str)	
	# parser.add_argument("--model_save_to", type=str, default="my-model", help='Name to save the file to')
	# parser.add_argument("--model_load_from", type=str, default="", help='Name to load the model from')
	# parser.add_argument("--num_layers", type=int, default=1, help='Int: Number of layers (1)')



	args = parser.parse_args()
	dict = vars(args)

	for i in dict:
		if (dict[i]=="False"):
			dict[i] = False
		elif dict[i]=="True":
			dict[i] = True
		
	kwargs = {	
				'model': dict['model'],
				'T':dict['T'],
				'n_epochs': dict['n_epochs'],
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
			  	'islin': dict['islin']
			}
	print(kwargs)
	main(**kwargs)
