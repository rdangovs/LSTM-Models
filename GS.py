import numpy as np 
import tensorflow as tf 
import timeit

from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest
from tensorflow.contrib.layers.python.layers.layers import batch_norm

def weight_variable(shape): 
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def gram_schmidt_columns(X):
    Q, R = np.linalg.qr(X)
    return Q

def gram_schmidt(vectors):
    basis = []
    for v in vectors:
        w = v - np.sum( np.dot(v,b)*b  for b in basis )
        if (w > 1e-10).any():  
            basis.append(w/np.linalg.norm(w))
    return np.array(basis)

def gram_schmidt_tensors (v_list):
    b_list = [tf.nn.l2_normalize(v_list[0], 1)]
    print(len(b_list))
    input()
    for i in range(1, len(v_list)):
        w = v_list[i] - tf.add_n([tf.reduce_sum(v_list[i] * b) * b for b in b_list])
        b_list.append(tf.nn.l2_normalize(w, 1))
    
    return tf.stack(b_list)

def normalize(x):
    print(x)
    """
    def f1(): return tf.constant([True])
    def f2(): return tf.constant([False])

    #ans = tf.cond(tf.less(x,tf.constant([0.1])),f1,f2)
    x1 = tf.constant(2)
    y1 = tf.constant(5)
    def f1(): return tf.multiply(x1, 17)
    def f2(): return tf.add(y1, 23)
    print(tf.less(x1,y1))
    print(tf.less(x,tf.constant(0.1)))
    input()
    tf.cond(tf.less(x,tf.constant(0.1)), f1, f2)
    r = tf.cond(tf.less(x, y), f1, f2)
    """
    return tf.nn.l2_normalize(x,0)

def rotation(x, y, size_batch, hidden_size):
    """computes the rotation between the two vectors u and v"""
    
    #get the sin/cos rotation matrix
    step1 = tf.reduce_sum(x * y, 1)
    step2 = tf.reduce_sum(x ** 2, 1) * tf.reduce_sum(y ** 2, 1)
    step3 = tf.sqrt(step2)
    costh = step1 / step3
    sinth = tf.sqrt(1 - costh ** 2)
    step4 = tf.reshape(costh, [size_batch, 1])
    step5 = tf.reshape(sinth, [size_batch, 1])
    step6 = tf.concat([step4, -step5, step5, step4], axis = 1)
    Rth = tf.reshape(step6, [size_batch, 2, 2])

    #get the u and v vectors 
    u = x/tf.reshape(tf.transpose(tf.sqrt(tf.reduce_sum(x ** 2, 1))),[size_batch,1])
    step8 = y - tf.reshape(tf.reduce_sum(u * y, 1),[size_batch,1]) * u
    v = step8/tf.reshape(tf.transpose(tf.sqrt(tf.reduce_sum(step8 ** 2, 1))),[size_batch,1])

    #concatenate the two vectors 
    step9 = tf.reshape(u,[size_batch,1,hidden_size])
    step14 = tf.reshape(v,[size_batch,1,hidden_size])
    step15 = tf.concat([step9,step14], axis = 1)
    step16 = tf.transpose(step15,[0,2,1])

    #do the batch matmul 
    step10 = tf.reshape(u,[size_batch,hidden_size,1])
    step11 = tf.transpose(step10,[0,2,1])
    uuT = tf.matmul(step10,step11)
    step12 = tf.reshape(v,[size_batch,hidden_size,1])
    step13 = tf.transpose(step12,[0,2,1])
    vvT = tf.matmul(step12,step13)
    
    #put all together 
    I = tf.eye(hidden_size, batch_shape=[size_batch])
    step17 = tf.matmul(tf.matmul(step16,Rth),step15)
    res = I - uuT - vvT - step17 

    return res 

def gs(X, row_vecs=True, norm = True):
    if not row_vecs:
        X = X.T
    Y = X[0:1,:].copy()
    for i in range(1, X.shape[0]):
        proj = np.diag((X[i,:].dot(Y.T)/np.linalg.norm(Y,axis=1)**2).flat).dot(Y)
        Y = np.vstack((Y, X[i,:] - proj.sum(0)))
    if norm:
        Y = np.diag(1/np.linalg.norm(Y,axis=1)).dot(Y)
    if row_vecs:
        return Y
    else:
        return Y.T

def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped

#Test numpy GS efficiency
"""
a = np.array([[1,2,5,10,17],[1,10,5,10,10],[1,2,10,10,17],[8,2,5,10,17],[10,2,5,10,17]])
wrapped = wrapper(gram_schmidt_columns, a)
print("First suggestion is ", timeit.timeit(wrapped, number=1000))
wrapped = wrapper(gram_schmidt, a)
print("Second suggestion is ", timeit.timeit(wrapped, number=1000))
wrapped = wrapper(gs, a)
print("Third suggestion is ", timeit.timeit(wrapped, number=1000))

print(gram_schmidt_columns(a))
print(gram_schmidt(a))
"""

#test QR on tensorflow
"""
W = weight_variable([5, 5])
Q, R = tf.qr(W)

c = tf.constant([[1.0,2.0,5.2],[3.0,4.0,1.2],[1.0,4.2,3.3]])
d = tf.constant([1.0,0.0,3.0])
b = tf.constant([7.0,8.0,7.6])
e = c * d + b
q, r = tf.qr(e)
sess = tf.Session()
result = sess.run(q)
print(result)
"""

#test basic math_ops operations 
#a = [tf.constant([[1.0, 2.0, 3.0],[1.0, 2.0, 3.0]]),tf.constant([[1.0, 5.0, 7.2],[1.0, 5.0, 7.2]]),tf.constant([[15.0, 2.8, 4.13],[15.0, 2.8, 4.13]])]
#ans = gram_schmidt_tensors(a)

a = tf.constant([[1.0,2.0,3.0,4.0,7.0]])
b = tf.constant([[5.0,3.0,6.0,7.0,2.0]])
c = rotation(a, b, 1, 5)


sess = tf.Session()
result = sess.run(c)
print(result)
