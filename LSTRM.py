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

def _rotation(x, 
			  y, 
			  size_batch,
			  hidden_size,
			  eps = 1e-12): 
	"""Rotation between two tensors: U(x,y) is unitary and takes x to y. 
	
	Args: 
		x: a tensor from where we want to start 
		y: a tensor at which we want to finish 
		eps: the cutoff for the normalizations (avoiding division by zero)
		size_batch: the size of the batch 
		hidden_size: the hidden size 

	Returns: 
		a tensor, which is the unitary rotation matrix U(x,y)
	"""

	#construct the 2x2 rotation
	u = tf.nn.l2_normalize(x, 1, epsilon = eps)
	costh = tf.reduce_sum(u * tf.nn.l2_normalize(y, 1, epsilon = eps), 1)
	sinth = tf.sqrt(1 - costh ** 2)
	step1 = tf.reshape(costh, [size_batch, 1])
	step2 = tf.reshape(sinth, [size_batch, 1])
	Rth = tf.reshape(tf.concat([step1, -step2, step2, step1], axis = 1), [size_batch, 2, 2])

	#get v and concatenate u and v 
	v = tf.nn.l2_normalize(y - tf.reshape(tf.reduce_sum(u * y, 1), [size_batch,1]) * u, 1, epsilon = eps)
	step3 = tf.concat([tf.reshape(u, [size_batch, 1, hidden_size]),
					  tf.reshape(v, [size_batch, 1, hidden_size])], 
					  axis = 1)
	
	#do the batch matmul 
	step4 = tf.reshape(u, [size_batch, hidden_size, 1])
	step5 = tf.reshape(v, [size_batch, hidden_size, 1])
	
	return (tf.eye(hidden_size, batch_shape = [size_batch]) - 
		   tf.matmul(step4, tf.transpose(step4, [0,2,1])) - 
		   tf.matmul(step5, tf.transpose(step5, [0,2,1])) - 
		   tf.matmul(tf.matmul(tf.transpose(step3, [0,2,1]), Rth), step3))

class BasicLSTRMCell(RNNCell):
	"""
	The Basic LSTRotationalM Cell. This model combines two ideas: 
											1) rotational (as a subclass of) unitary memory. 
											2) utility for matrix memory states.
	"""
	def __init__(self, hidden_size, forget_bias = 1.0, 
				 activation = None, size_batch = 128, reuse = None, isMatrix = False, isActivation = False): 
		super(BasicLSTRMCell, self).__init__(_reuse = reuse)
		self._hidden_size = hidden_size 
		self._forget_bias = forget_bias 
		self._size_batch = size_batch
		self._activation = activation or relu
		self._isMatrix = isMatrix
		self._isActivation = isActivation

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
			C = tf.reshape(state[0],[self._size_batch, self._hidden_size, self._hidden_size])
		else:
			c = state[0]

		concat = _linear([inputs, state[1]], 4 * self._hidden_size, True)
		# i = input_gate, j = new_input, f = forget_gate, o = output_gate
		i, j, f, o = array_ops.split(value = concat, num_or_size_splits = 4, axis = 1)

		if self._isMatrix:
			d = tf.reshape(tf.matmul(C, tf.reshape(sigmoid(i) * tanh(j),[self._size_batch,self._hidden_size,1])),[self._size_batch,self._hidden_size])
		else: 
			d = sigmoid(i) * tanh(j)

		#get the rotation matrix from f to d
		U = _rotation(f, d, self._size_batch, self._hidden_size)

		#put all together 
		if self._isMatrix:
			o = tf.reshape(o,[self._size_batch, self._hidden_size, 1])
			new_h = tf.reshape(tf.matmul(self._activation(U), o), [self._size_batch, self._hidden_size])
			if self._isActivation:
				new_c = self._activation(tf.reshape(U, [self._size_batch, self._hidden_size ** 2]))
			else:
				new_c = tf.reshape(U, [self._size_batch, self._hidden_size ** 2])
		else: 
			if self._isActivation:
				new_c = self._activation(tf.reshape(tf.matmul(U, tf.reshape(c,[self._size_batch,self._hidden_size,1])), [self._size_batch, self._hidden_size]))
				new_h = new_c * o
			else:
				new_c = tf.reshape(tf.matmul(U, tf.reshape(c,[self._size_batch,self._hidden_size,1])), [self._size_batch, self._hidden_size])
				new_h = self._activation(new_c * o)

		new_state = LSTMStateTuple(new_c, new_h)

		return new_h, new_state
