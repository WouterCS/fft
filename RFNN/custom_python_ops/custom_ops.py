import numpy as np
import tensorflow as tf
#from op_creator import custom_wih_grad

# custom reimplementation of tf.sigmoid
def tf_sigmoid(xin):
    # numpy implementation of the desired function
    def custom_op(x):
        xout = 1 / (1 + np.exp(-x))
        return (xout).astype(np.float32)

    # tensorflow implementation of the new gradient as it is propogated through this op
    def custom_op_grad(op, grad):
        x = op.inputs[0]
        sigmoidX = (1 / (1 + tf.exp(-x)))
        return grad * (sigmoidX *(1 - sigmoidX))
    
    # custom_wih_grad takes a list of the arguments of the function, the numpy implementation of the function, the gradient implementation and the shape of the output.
    return custom_wih_grad([xin], custom_op, custom_op_grad, xin.shape)

# custom reimplementation of tf.abs
def tf_abs(xin):
    # numpy implementation of the desired function
    def custom_op(x):
        xout = np.abs(x)
        return (xout).astype(np.float32)

    # tensorflow implementation of the new gradient as it is propogated through this op
    def custom_op_grad(op, grad):
        x = op.inputs[0]
        return tf.sign(x) * grad
    
    # custom_wih_grad takes a list of the arguments of the function, the numpy implementation of the function, the gradient implementation and the shape of the output.
    return custom_wih_grad([xin], custom_op, custom_op_grad, xin.shape)

# custom reimplementation of tf.abs
def tf_relu(xin):
    # numpy implementation of the desired function
    def custom_op(x):
        xout = np.maximum(x, 0)
        return (xout).astype(np.float32)

    # tensorflow implementation of the new gradient as it is propogated through this op
    def custom_op_grad(op, grad):
        x = op.inputs[0]
        return ((tf.sign(x) + 1) / 2) * grad
    
    # custom_wih_grad takes a list of the arguments of the function, the numpy implementation of the function, the gradient implementation and the shape of the output.
    return custom_wih_grad([xin], custom_op, custom_op_grad, xin.shape)

# custom tensorflow implementation of np.arctan2
def tf_arctan2(yin, xin):
    # numpy implementation of the desired function
    def custom_op(y, x):
        xout = np.arctan2(y, x)
        return (xout).astype(np.float32)

    # tensorflow implementation of the new gradient as it is propogated through this op
    def custom_op_grad(op, grad):
        y = op.inputs[0]
        x = op.inputs[1]
        
        return (grad * x / (tf.square(x) + tf.square(y)), grad * -y / (tf.square(x) + tf.square(y)))
    
    # custom_wih_grad takes a list of the arguments of the function, the numpy implementation of the function, the gradient implementation and the shape of the output.
    return custom_wih_grad([yin, xin], custom_op, custom_op_grad, xin.shape)

