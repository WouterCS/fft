import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np

def np_atan2(x,y):
    return np.arctan2(x, y).astype(np.float32)

def atan2grad(op, grad):
    y = op.inputs[0]
    x = op.inputs[1]
    
    if x == 0 and y == 0:
        return 0, 0
    
    return grad * (x / (tf.square(x) + tf.square(y))), grad * (y /  (tf.square(x) + tf.square(y))) #the propagated gradient with respect to the first and second argument respectively

def py_func(func, inp, Tout, stateful=True, name=None, grad=None):

    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)


def tf_atan2(x,y, name=None):

    with ops.name_scope("atan2", name, [x,y]) as name:
        z = py_func(np_atan2,
                        [x,y],
                        [tf.float32],
                        name=name,
                        grad=atan2grad)
        z= tf.reshape(tf.concat(z,1), x.shape)
        return z
    
def tf_angle(c):
    return tf_atan2(tf.imag(c), tf.real(c))

def sqrtMagnitude(c):
    mag = tf.cast(tf.abs(c), tf.float32)
    pha = tf.cast(tf_angle(c), tf.float32)
    
    mag = tf.sqrt(mag)
    
    magComplex = tf.complex(mag, tf.zeros(mag.shape))
    phaComplex = tf.complex(tf.zeros(pha.shape), pha)
    return magComplex * tf.exp( phaComplex )