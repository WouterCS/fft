import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np

# wrapper for the numpy arctan2 function, where we make sure the output is of type float32
def custom_op(x):
    return (1 / (1 + np.exp(-x))).astype(np.float32)

# Function that can propegate the gradient through an arctan2 operation, in the format expected by Tensorflow
def custom_op_grad(op, grad):
    x = op.inputs[0]
    
    sigmoidX = (1 / (1 + tf.exp(-x)))
    return grad * (sigmoidX *(1 - sigmoidX))
    
# Wrapper around tf.py_func which allows gradients to be added.
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

# Tensorflow op that combines the arctan2 wrapper and the gradient function using  py_func to create a tensorflow op that performs the arctan2 function on pairs of tensors. 
def tf_custom_op(x, name=None):

    with ops.name_scope("atan2", name, [x]) as name:
        z = py_func(custom_op,
                        [x],
                        [tf.float32],
                        name=name,
                        grad=custom_op_grad)
        # This reshape is necessary to ensure that the output has a known shape.
        z= tf.reshape(tf.concat(z,1), x.shape)
        return z
    
# Uses the arctan2 tensorflow op defined in this file to create an angle function, working on complex tensors.
def tf_sigmoid(x):
    return tf_custom_op(x)