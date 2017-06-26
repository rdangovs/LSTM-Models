#Adapted from `https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/python/ops/rnn_cell_impl.py`

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import hashlib
import numbers
import tensorflow as tf

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest

from tensorflow.python.ops.rnn_cell_impl import RNNCell
from tensorflow.python.ops.rnn_cell_impl import _linear
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple
from tensorflow.python.ops import control_flow_ops

sigmoid = math_ops.sigmoid 
tanh = math_ops.tanh
matm = math_ops.matmul
mul = math_ops.multiply 
relu = nn_ops.relu

class BasicLSTUMCell(RNNCell):
	def __init__(self, hidden_size, forget_bias = 1.0, 
				 activation = None, size_batch = 128, reuse = None): 
		super(BasicLSTUMCell, self).__init__(_reuse = reuse)
		self._hidden_size = hidden_size 
		self._forget_bias = forget_bias 
		self._size_batch = size_batch
		self._activation = activation or relu

	@property 
	def state_size(self):
		return (LSTMStateTuple(self._hidden_size, self._hidden_size))

	@property 
	def output_size(self): 
		return self._hidden_size


	def call(self, inputs, state): 
		"""	
		Long short-term unitary memory cell (LSTUM).
		"""
		c, h = state
		C = tf.reshape(c,[self._size_batch, self._hidden_size, self._hidden_size])

		concat = _linear([inputs, h], 4 * self._hidden_size, True)
		# i = input_gate, j = new_input, f = forget_gate, o = output_gate
		i, j, f, o = array_ops.split(value = concat, num_or_size_splits = 4, axis = 1)
		
		d = sigmoid(i) * tanh(j)
		e = tf.multiply(C, tf.reshape(f,[self._size_batch,1,self._hidden_size,])) + tf.reshape(d,[self._size_batch,1,self._hidden_size])
		e_l = tf.Variable(tf.unstack(e, axis = 2)) # 128 128 128 
		bList = tf.Variable([tf.nn.l2_normalize(e_l[0], 1)]) # 1 128 128  
		print(type(tf.shape(bList)[0]))
		#Gram-Schmidt loop
		i = tf.constant(0)
		loop_vars = [bList, i]
		shape_inv=[tf.TensorShape([None, self._size_batch, self._hidden_size]), i.get_shape()]
		cond = lambda b_l, i: tf.less(i, self._size_batch)
		def F(b_l, i): 
			TensorArr = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False) 
			array = TensorArr.unstack(b_l)
			print(array.read(0))

			b_u = tf.unstack(tf.reshape(b_l,[tf.shape(b_l)[0],self._size_batch, self._hidden_size]))
			input()

			b_u = tf.unstack(tf.reshape(b_l,[i+1,self._size_batch, self._hidden_size]))
			dot = b_l[i] * b_u
			reduce_dot_prime = tf.reduce_sum(dot, axis = 2)
			reduce_dot_final = tf.reduce_sum(b_l * reduce_dot_prime, axis = 0)
			
			w_n = e_l[i] - reduce_dot_final 
			w_n = tf.nn.l2_normalize(w_n, 1, epsilon=1e-8)
			b_l = tf.concat([b_l, tf.reshape(w_n, [1,self._size_batch,self._hidden_size])], 0)
			return b_l, i

		b_list, _ = control_flow_ops.while_loop(cond, F, loop_vars, shape_inv)

		print(b_list)
		input



		new_C = tf.stack(b_list, axis=1)

		o = tf.reshape(o,[self._size_batch, self._hidden_size, 1])
		new_h = tf.matmul(self._activation(new_C), o)
		new_h = tf.reshape(new_h, [self._size_batch, self._hidden_size])
		new_c = tf.reshape(new_C, [self._size_batch, self._hidden_size ** 2])
		new_state = LSTMStateTuple(new_c, new_h)

		return new_h, new_state

		



