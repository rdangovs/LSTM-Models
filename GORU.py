from tensorflow.python.ops.rnn_cell_impl import RNNCell

from EUNN import *

# def lrelu(x, leak=0.2, name="lrelu"):
#      with tf.variable_scope(name):
#          f1 = 0.5 * (1 + leak)
#          f2 = 0.5 * (1 - leak)
#          return f1 * x + f2 * abs(x)

# def clip_relu(z, max_value=1):
# 	step1 = nn_ops.relu(z)
# 	step2 = max_value - nn_ops.relu(max_value - step1)
# 	return step2

# def clip_modReLU(z, b, comp):
# 	if comp:
# 		z_norm = math_ops.sqrt(math_ops.square(math_ops.real(z)) + math_ops.square(math_ops.imag(z))) + 0.00001
# 		# step1 = clip_ops.clip_by_value(nn_ops.bias_add(z_norm, b), 0, 5)
# 		step1 = nn_ops.bias_add(z_norm, b)
# 		step2 = math_ops.complex(clip_relu(step1), array_ops.zeros_like(z_norm))
# 		step3 = math_ops.complex(math_ops.real(z)/z_norm, math_ops.imag(z)/z_norm)
# 		# step3 = z/math_ops.complex(z_norm, array_ops.zeros_like(z_norm))
# 	else:
# 		z_norm = math_ops.abs(z) + 0.00001
# 		step1 = nn_ops.bias_add(z_norm, b)
# 		step2 = clip_relu(step1)
# 		step3 = math_ops.sign(z)
		
# 	return math_ops.multiply(step3, step2)	


# def modReLU(z, b, comp):
# 	if comp:
# 		z_norm = math_ops.sqrt(math_ops.square(math_ops.real(z)) + math_ops.square(math_ops.imag(z))) + 0.00001
# 		# step1 = clip_ops.clip_by_value(nn_ops.bias_add(z_norm, b), 0, 5)
# 		step1 = nn_ops.bias_add(z_norm, b)
# 		step2 = math_ops.complex(nn_ops.relu(step1), array_ops.zeros_like(z_norm))
# 		step3 = math_ops.complex(math_ops.real(z)/z_norm, math_ops.imag(z)/z_norm)
# 		# step3 = z/math_ops.complex(z_norm, array_ops.zeros_like(z_norm))
# 	else:
# 		z_norm = math_ops.abs(z) + 0.00001
# 		step1 = nn_ops.bias_add(z_norm, b)
# 		step2 = nn_ops.relu(step1)
# 		step3 = math_ops.sign(z)
		
# 	return math_ops.multiply(step3, step2)

# def mod_sigmoid(z):
# 	z_norm = math_ops.sqrt(math_ops.square(math_ops.real(z)) + math_ops.square(math_ops.imag(z))) + 0.00001
# 	# step1 = nn_ops.bias_add(z_norm, b)
# 	step1 = z_norm
# 	step2 = math_ops.complex(math_ops.sigmoid(step1), array_ops.zeros_like(z_norm))
# 	step3 = math_ops.complex(math_ops.real(z)/z_norm, math_ops.imag(z)/z_norm)

	# return math_ops.multiply(step3, step2)

def modReLU(z, b):
	z_norm = math_ops.abs(z) + 0.00001
	step1 = nn_ops.bias_add(z_norm, b)
	step2 = nn_ops.relu(step1)
	step3 = math_ops.sign(z)		
	return math_ops.multiply(step3, step2)




class GORUCell(RNNCell):


	def __init__(self, hidden_size, capacity=2, FFT=False, activation=modReLU):
		
		self._hidden_size = hidden_size
		self._activation = activation
		self._capacity = capacity
		self._FFT = FFT
		# self._comp = comp

		self.v1, self.v2, self.ind, _, self._capacity = EUNN_param(hidden_size, capacity, FFT, False)



	@property
	def state_size(self):
		return self._hidden_size

	@property
	def output_size(self):
		return self._hidden_size

	@property
	def capacity(self):
		return self._capacity

	def __call__(self, inputs, state, scope=None):
		with vs.variable_scope(scope or "GORU_cell"):

			# U_r = vs.get_variable("U_r", [inputs.get_shape()[-1], self._hidden_size], dtype=tf.float32)
			# U_rx = math_ops.matmul(inputs, U_r) 
			# U_g = vs.get_variable("U_g", [inputs.get_shape()[-1], self._hidden_size], dtype=tf.float32)
			# U_gx = math_ops.matmul(inputs, U_g) 
			# # U_c = vs.get_variable("U_c", [inputs.get_shape()[-1], self._hidden_size], dtype=tf.float32)
			# # U_cx = math_ops.matmul(inputs, U_c) 
			# if self._comp:
			# 	U_c_re = vs.get_variable("U_c_re", [inputs.get_shape()[-1], self._hidden_size], initializer=init_ops.random_uniform_initializer(-0.01, 0.01))
			# 	U_c_im = vs.get_variable("U_c_im", [inputs.get_shape()[-1], self._hidden_size], initializer=init_ops.random_uniform_initializer(-0.01, 0.01))
			# 	U_cx_re = math_ops.matmul(inputs, U_c_re)
			# 	U_cx_im = math_ops.matmul(inputs, U_c_im)
			# 	U_cx = math_ops.complex(U_cx_re, U_cx_im)
			# 	U_rx = math_ops.complex(U_rx, array_ops.zeros_like(U_rx))
			# 	U_gx = math_ops.complex(U_gx, array_ops.zeros_like(U_gx))
			# else:
			# 	U_c = vs.get_variable("U_c", [inputs.get_shape()[-1], self._hidden_size], initializer=init_ops.random_uniform_initializer(-0.01, 0.01))
			# 	U_cx = math_ops.matmul(inputs, U_c) 
			# if self._comp:
			# 	U_re = vs.get_variable("U_re", [inputs.get_shape()[-1], self._hidden_size * 3], dtype=tf.float32)
			# 	U_im = vs.get_variable("U_im", [inputs.get_shape()[-1], self._hidden_size * 3], dtype=tf.float32)
			# 	Ux_re = math_ops.matmul(inputs, U_re)
			# 	Ux_im = math_ops.matmul(inputs, U_im)
			# 	Ux = math_ops.complex(Ux_re, Ux_im)
			# 	U_cx, U_rx, U_gx = array_ops.split(Ux, 3, axis=1)
			# else:
			U = vs.get_variable("U", [inputs.get_shape()[-1], self._hidden_size * 3], dtype=tf.float32, initializer=init_ops.random_uniform_initializer(-0.01, 0.01))
			Ux = math_ops.matmul(inputs, U)
			U_cx, U_rx, U_gx = array_ops.split(Ux, 3, axis=1)


			# if self._comp:
			# 	W_r_re = vs.get_variable("W_r_re", [self._hidden_size, self._hidden_size], dtype=tf.float32)
			# 	W_g_re = vs.get_variable("W_g_re", [self._hidden_size, self._hidden_size], dtype=tf.float32)
			# 	W_r_im = vs.get_variable("W_r_im", [self._hidden_size, self._hidden_size], dtype=tf.float32)
			# 	W_g_im = vs.get_variable("W_g_im", [self._hidden_size, self._hidden_size], dtype=tf.float32)
			# 	W_r = math_ops.complex(W_r_re, W_r_im)
			# 	W_g = math_ops.complex(W_g_re, W_g_im)
			# 	W_rh = math_ops.matmul(state, W_r)
			# 	W_gh = math_ops.matmul(state, W_g)
			# else:
			W_r = vs.get_variable("W_r", [self._hidden_size, self._hidden_size], dtype=tf.float32, initializer=init_ops.random_uniform_initializer(-0.01, 0.01))
			W_g = vs.get_variable("W_g", [self._hidden_size, self._hidden_size], dtype=tf.float32, initializer=init_ops.random_uniform_initializer(-0.01, 0.01))
			W_rh = math_ops.matmul(state, W_r)
			W_gh = math_ops.matmul(state, W_g)

			
			
			# if self._comp:
			# 	bias_r_re = vs.get_variable("bias_r_re", [self._hidden_size], dtype=tf.float32, initializer=init_ops.constant_initializer(2.))
			# 	bias_g_re = vs.get_variable("bias_g_re", [self._hidden_size], dtype=tf.float32)
			# 	bias_r_im = vs.get_variable("bias_r_im", [self._hidden_size], dtype=tf.float32, initializer=init_ops.constant_initializer(0.))
			# 	bias_g_im = vs.get_variable("bias_g_im", [self._hidden_size], dtype=tf.float32)
			# 	bias_c = vs.get_variable("bias_c", [self._hidden_size], dtype=tf.float32, initializer=init_ops.constant_initializer(0.))
			# 	bias_r = math_ops.complex(bias_r_re, bias_r_im)
			# 	bias_g = math_ops.complex(bias_g_re, bias_g_im)
			# 	# bias_r = vs.get_variable("bias_r", [self._hidden_size], dtype=tf.float32, initializer=init_ops.constant_initializer(2.))
			# 	# bias_g = vs.get_variable("bias_g", [self._hidden_size], dtype=tf.float32)
			# 	# bias_c = vs.get_variable("bias_c", [self._hidden_size], dtype=tf.float32, initializer=init_ops.constant_initializer(0.))
		
			# else:
			bias_r = vs.get_variable("bias_r", [self._hidden_size], dtype=tf.float32, initializer=init_ops.constant_initializer(2.))
			bias_g = vs.get_variable("bias_g", [self._hidden_size], dtype=tf.float32)
			bias_c = vs.get_variable("bias_c", [self._hidden_size], dtype=tf.float32, initializer=init_ops.constant_initializer(2.))
		





			# def norm(z):
			# 	return math_ops.sqrt(math_ops.square(math_ops.real(z)) + math_ops.square(math_ops.imag(z)))
			
			# def phase(z, norm_z):
			# 	return math_ops.complex(math_ops.real(z)/norm_z, math_ops.imag(z)/norm_z)

			# if self._comp:
			# 	# r_tmp = U_rx + W_rh + bias_r
			# 	# g_tmp = U_gx + W_gh + bias_g
			# 	# r = math_ops.complex(math_ops.sigmoid(math_ops.real(r_tmp)), math_ops.sigmoid(math_ops.imag(r_tmp)))
			# 	# g = math_ops.complex(math_ops.sigmoid(math_ops.real(g_tmp)), math_ops.sigmoid(math_ops.imag(g_tmp)))

			# 	# REAL PART APPROACH
			# 	# r = math_ops.sigmoid(math_ops.real(r_tmp))
			# 	# g = math_ops.sigmoid(math_ops.real(g_tmp))
			# 	# o = 1 - g

			# 	# r = math_ops.complex(r, array_ops.zeros_like(r))
			# 	# g = math_ops.complex(g, array_ops.zeros_like(g))
			# 	# o = math_ops.complex(o, array_ops.zeros_like(o))

			# 	# MOD SIGMOID APPROACH
			# 	r_tmp = U_rx + W_rh + bias_r
			# 	g_tmp = U_gx + W_gh + bias_g
			# 	# r = math_ops.complex(math_ops.sigmoid(math_ops.real(r_tmp)), math_ops.sigmoid(math_ops.imag(r_tmp)))
			# 	# g = math_ops.complex(math_ops.sigmoid(math_ops.real(g_tmp)), math_ops.sigmoid(math_ops.imag(g_tmp)))

			# 	r = mod_sigmoid(r_tmp)
			# 	g = mod_sigmoid(g_tmp)
			# 	# r = U_rx + W_rh
			# 	# g = U_gx + W_gh

			# 	norm_g = norm(g)
			# 	o = math_ops.complex((1 - norm_g), array_ops.zeros_like(norm_g)) * phase(g, norm_g)
			
			# else:
			r_tmp = U_rx + W_rh + bias_r
			g_tmp = U_gx + W_gh + bias_g
			r = math_ops.sigmoid(r_tmp)
			# g = vs.get_variable("fix_gate", [self._hidden_size], dtype=tf.float32, initializer=init_ops.constant_initializer(0.5))


			g = math_ops.sigmoid(g_tmp)

			o = 1- g

			# Unitaryh = EUNN_loop(math_ops.multiply(r, state), self._capacity, self.v1, self.v2, self.ind, self.diag)
			# c = modReLU(Unitaryh + U_cx, bias_c, self._comp)

			Unitaryh = EUNN_loop(state, self._capacity, self.v1, self.v2, self.ind, None)
			c = modReLU(math_ops.multiply(r, Unitaryh) + U_cx, bias_c)
			new_state = math_ops.multiply(g, state) +  math_ops.multiply(o, c)

		return new_state, new_state

