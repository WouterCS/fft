import numpy as np
import tensorflow as tf
from custom_ops import tf_arctan2


def tf_angle(c):
    arctanFun = tf_arctan2()
    return arctanFun(tf.imag(c), tf.real(c))

def sqrtMagnitude(c):
    mag = tf.abs(c)
    pha = tf_angle(c)
    
    sqrtmag = tf.sqrt(mag)
    
    magCompl = tf.complex(sqrtmag, tf.zeros(sqrtmag.shape))
    phaCompl = tf.complex(tf.zeros(pha.shape), pha)
    
    return magCompl * tf.exp(phaCompl)