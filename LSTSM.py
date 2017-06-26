#Adapted from `https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/python/ops/rnn_cell_impl.py`

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import hashlib
import numbers

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

sigmoid = math_ops.sigmoid 
tanh = math_ops.tanh
matm = math_ops.matmul
mul = math_ops.multiply 
relu = nn_ops.relu

class BasicLSTSMCell(RNNCell):
	#note that state is always a tuple
	def __init__(self, hidden_size, forget_bias = 1.0, 
				 activation = None, reuse = None): 
		super(BasicLSTSMCell, self).__init__(_reuse = reuse)
		self._hidden_size = hidden_size 
		self._forget_bias = forget_bias 
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
		Note that the c-vector is fixed.
		"""
		c, h = state 
		concat = _linear([inputs, h], 4 * self._hidden_size, True)
		# i = input_gate, j = new_input, f = forget_gate, o = output_gate
		i, j, f, o = array_ops.split(value = concat, num_or_size_splits = 4, axis = 1)

		temp_c = c * sigmoid(f + self._forget_bias) + sigmoid(i) * tanh(j)
		new_h = self._activation(temp_c) * sigmoid(o)
		new_state = LSTMStateTuple(c, new_h)

		return new_h, new_state
		



