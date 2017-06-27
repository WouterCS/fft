import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

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
    
