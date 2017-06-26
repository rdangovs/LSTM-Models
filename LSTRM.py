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
from tensorflow.python.ops.control_flow_ops import cond

sigmoid = math_ops.sigmoid 
tanh = math_ops.tanh
matm = math_ops.matmul
mul = math_ops.multiply 
relu = nn_ops.relu

class BasicLSTRMCell(RNNCell):
	"""
	The Basic LSTRotationalM Cell. This model combines two ideas: 
											1) rotational (as a subclass of) unitary memory. 
											2) utility for matrix memory states.
	"""
	def __init__(self, hidden_size, forget_bias = 1.0, 
				 activation = None, size_batch = 128, reuse = None, isMatrix = None): 
		super(BasicLSTRMCell, self).__init__(_reuse = reuse)
		self._hidden_size = hidden_size 
		self._forget_bias = forget_bias 
		self._size_batch = size_batch
		self._activation = activation or relu
		self._isMatrix = isMatrix

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

		if self._isMatrix: 
			c, h = state
			C = tf.reshape(c,[self._size_batch, self._hidden_size, self._hidden_size])
		else:
			c, h = state

		concat = _linear([inputs, h], 4 * self._hidden_size, True)
		# i = input_gate, j = new_input, f = forget_gate, o = output_gate
		i, j, f, o = array_ops.split(value = concat, num_or_size_splits = 4, axis = 1)
		
		d = sigmoid(i) * tanh(j)

		if self._isMatrix: 
			d_temp = tf.matmul(C, tf.reshape(d,[self._size_batch,self._hidden_size,1]))
			d = tf.reshape(d_temp,[self._size_batch,self._hidden_size])

		#get the rotation matrix from f to d
		step1 = tf.nn.l2_normalize(f, 1, epsilon=1e-8)
		step2 = tf.nn.l2_normalize(d, 1, epsilon=1e-8)
		costh = tf.reduce_sum(step1 * step2, 1)

		sinth = tf.sqrt(1 - costh ** 2)
		step4 = tf.reshape(costh, [self._size_batch, 1])
		step5 = tf.reshape(sinth, [self._size_batch, 1])
		step6 = tf.concat([step4, -step5, step5, step4], axis = 1)
		Rth = tf.reshape(step6, [self._size_batch, 2, 2])
		
		#get the u and v vectors 
		u = step1 
		step8 = d - tf.reshape(tf.reduce_sum(u * d, 1),[self._size_batch,1]) * u
		v = tf.nn.l2_normalize(step8, 1, epsilon=1e-8)

		#concatenate the two vectors 
		step9 = tf.reshape(u,[self._size_batch,1,self._hidden_size])
		step14 = tf.reshape(v,[self._size_batch,1,self._hidden_size])
		step15 = tf.concat([step9,step14], axis = 1)
		step16 = tf.transpose(step15,[0,2,1])

		#do the batch matmul 
		step10 = tf.reshape(u,[self._size_batch,self._hidden_size,1])
		step11 = tf.transpose(step10,[0,2,1])
		uuT = tf.matmul(step10,step11)
		step12 = tf.reshape(v,[self._size_batch,self._hidden_size,1])
		step13 = tf.transpose(step12,[0,2,1])
		vvT = tf.matmul(step12,step13)
		
		#put all together 
		I = tf.eye(self._hidden_size, batch_shape=[self._size_batch])
		step17 = tf.matmul(tf.matmul(step16,Rth),step15)
		res = I - uuT - vvT - step17

		if self._isMatrix:
			new_C = res
			o = tf.reshape(o,[self._size_batch, self._hidden_size, 1])
			new_h = tf.matmul(self._activation(new_C), o)
			new_h = tf.reshape(new_h, [self._size_batch, self._hidden_size])
			new_c = tf.reshape(new_C, [self._size_batch, self._hidden_size ** 2])
		else: 
			new_c = tf.reshape(tf.matmul(res, tf.reshape(c,[self._size_batch,self._hidden_size,1])), [self._size_batch, self._hidden_size])	
			new_h = self._activation(new_c) * o 

		new_state = LSTMStateTuple(new_c, new_h)

		return new_h, new_state
