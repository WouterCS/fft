import tensorflow as tf
import numpy as np

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var
  

def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var
  

def cifar10_example_inference(images, weights, params):
  """Build the CIFAR-10 model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  batchsize = images.shape[0].value
  
  #with tf.variable_scope('conv1') as scope:
  print('model active')
  kernel = weights['layer1']['kernel']
  conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
  biases1 = weights['layer1']['biases']
  pre_activation = tf.nn.bias_add(conv, biases1)
  conv1 = tf.nn.relu(pre_activation)#, name=scope.name)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  # conv2
  #with tf.variable_scope('conv2') as scope:
  kernel = weights['layer2']['kernel']
  conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
  biases2 = weights['layer2']['biases']
  pre_activation = tf.nn.bias_add(conv, biases2)
  conv2 = tf.nn.relu(pre_activation)#, name=scope.name)

  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # local3
  #with tf.variable_scope('local3') as scope:
  # Move everything into depth so we can perform a single matrix multiply.
  reshape = tf.reshape(pool2, [batchsize, -1])
  dim = reshape.get_shape()[1].value
  
  t = weights['layer3']['biases']
  
  weights = weights['layer3']['weights']
  t = weights['layer3']['biases']
  local3 = tf.nn.relu(tf.matmul(reshape, weights) + t)#, name=scope.name)

  # local4
  #with tf.variable_scope('local4') as scope:
  weights = weights['layer4']['weights']
  biases4 = weights['layer4']['biases']
  local4 = tf.nn.relu(tf.matmul(local3, weights) + biases4)#, name=scope.name)

  # linear layer(WX + b),
  # We don't apply softmax here because
  # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
  # and performs the softmax internally for efficiency.
  #with tf.variable_scope('softmax_linear') as scope:
  weights = weights['layer5']['weights']
  biases5 = weights['layer5']['biases']
  softmax_linear = tf.add(tf.matmul(local4, weights), biases5)#, name=scope.name)

  return softmax_linear
  
def tfCifarWeightsWeights(params, sizeFinalImage, dataset):
    layer1Weights = {'kernel': _variable_with_weight_decay('weights1',
                                       shape=[5, 5, 3, params.N1],
                                       stddev=5e-2,
                                       wd=0.0),
                    'biases': _variable_on_cpu('biases1', [params.N1], tf.constant_initializer(0.0))}
                    
    layer2Weights = {'kernel': _variable_with_weight_decay('weights2',
                                       shape=[5, 5, params.N1, params.N2],
                                       stddev=5e-2,
                                       wd=0.0),
                    'biases': _variable_on_cpu('biases2', [params.N2], tf.constant_initializer(0.1))}
    
    dim = (dataset['width'] / 4) * (dataset['height'] / 4) * params.N2
    layer3Weights = {'weights': _variable_with_weight_decay('weights3', shape=[dim, 384],
                                        stddev=0.04, wd=0.004),
                    'biases': _variable_on_cpu('biases3', [384], tf.constant_initializer(0.1))}
                    
    layer4Weights = {'weights': _variable_with_weight_decay('weights4', shape=[384, 192],
                                        stddev=0.04, wd=0.004),
                    'biases': _variable_on_cpu('biases4', [192], tf.constant_initializer(0.1))}
    
    layer5Weights = {'weights': _variable_with_weight_decay('weights5', [192, params.num_classes],
                                        stddev=1/192.0, wd=0.0),
                    'biases': _variable_on_cpu('biases5', [params.num_classes], tf.constant_initializer(0.0))}
                    
    weights = {'layer1': layer1Weights, 'layer2': layer2Weights, 'layer3': layer3Weights, 'layer4': layer4Weights, 'layer5': layer5Weights, }
    return weights
  
def train(total_loss, global_step, params):
  """Train CIFAR-10 model.
  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.
  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(float(params.initial_lr),
                                  global_step,
                                  params.max_epochs *( params.number_of_training_samples // params.batchsize ),
                                  params.min_lr,
                                  staircase=True)


  # Generate moving averages of all losses.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op
  
