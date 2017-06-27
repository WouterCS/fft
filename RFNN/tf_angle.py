import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np


    
# Wrapper around tf.py_func which allows gradients to be added.
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)
    
def custom_wih_grad(args, custom_op, custom_op_grad, outshape):
    with ops.name_scope("atan2", "MyOp", args) as name:
        z = py_func(custom_op,
                        args,
                        [tf.float32],
                        name=name,
                        grad=custom_op_grad)
        # This reshape is necessary to ensure that the output has a known shape.
        z= tf.reshape(tf.concat(z,1), outshape)
        return z
    
# Uses the arctan2 tensorflow op defined in this file to create an angle function, working on complex tensors.
def tf_sigmoid():
    # wrapper for the numpy arctan2 function, where we make sure the output is of type float32
    def custom_op(x):
        xout = 1 / (1 + np.exp(-x))
        return (xout).astype(np.float32)

    # Function that can propegate the gradient through an arctan2 operation, in the format expected by Tensorflow
    def custom_op_grad(op, grad):
        x = op.inputs[0]
        sigmoidX = (1 / (1 + tf.exp(-x)))
        return grad * (sigmoidX *(1 - sigmoidX))
    
    return lambda xin: custom_wih_grad([xin], custom_op, custom_op_grad, xin.shape)

def tf_abs():
    # wrapper for the numpy arctan2 function, where we make sure the output is of type float32
    def custom_op(x):
        xout = np.abs(x)
        return (xout).astype(np.float32)

    # Function that can propegate the gradient through an arctan2 operation, in the format expected by Tensorflow
    def custom_op_grad(op, grad):
        x = op.inputs[0]
        return tf.sign(x) * grad
    
    return lambda xin: custom_wih_grad([xin], custom_op, custom_op_grad, xin.shape)

# Uses the arctan2 tensorflow op defined in this file to create an angle function, working on complex tensors.
def tf_arctan2():
    # wrapper for the numpy arctan2 function, where we make sure the output is of type float32
    def custom_op(y, x):
        xout = np.arctan2(y, x)
        return (xout).astype(np.float32)

    # Function that can propegate the gradient through an arctan2 operation, in the format expected by Tensorflow
    def custom_op_grad(op, grad):
        y = op.inputs[0]
        x = op.inputs[1]
        
        return (grad * x / (tf.square(x) + tf.square(y)), grad * -y / (tf.square(x) + tf.square(y)))
    
    return lambda yin, xin: custom_wih_grad([yin, xin], custom_op, custom_op_grad, xin.shape)

def tf_angle(c):
    arctanFun = tf_arctan2()
    return arctanFun(tf.imag(c), tf.real(c))

def idThroughPolar(c):
    mag = tf.abs(c)
    pha = tf_angle(c)
    
    sqrtmag = tf.sqrt(mag)
    
    magCompl = tf.complex(sqrtmag, tf.zeros(sqrtmag.shape))
    phaCompl = tf.complex(tf.zeros(pha.shape), pha)
    
    return magCompl * tf.exp(phaCompl)