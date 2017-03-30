# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""All user ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
  
from tensorflow.python.ops import gen_user_ops
from tensorflow.python.ops.gen_user_ops import *

# ------------------------------------------
# Imports for GaussianBasisFilters operation
# ------------------------------------------
import os.path
from tensorflow.python.platform import resource_loader
from tensorflow.python.framework import load_library
from tensorflow.python.framework import ops

def my_fact():
  """Example of overriding the generated code for an Op."""
  return gen_user_ops._fact()

# ------------------------------------------
# Register GaussianBasisFilters operation
# ------------------------------------------

# Load module
_gaussian_basis_filters_module = load_library.load_op_library(os.path.join(resource_loader.get_data_files_path(), 'gaussian_basis_filters.so'))

# Register operation
gaussian_basis_filters = _gaussian_basis_filters_module.gaussian_basis_filters

"""
tensorflow.user_ops.gaussian_basis_filters(grid, order, sigma_x, sigma_y, normalize):

 Creates a filter bank that contains all the 2-dimensional gaussian derivatives up to a certain order.

 There are two methods to construct the filters:
  - discrete analog of the gaussian  (grid='discrete')
  - sampled continuous gaussian      (grid='continuous')

 The normalize parameter controls whether the filters are L1-norm normalized or not.

 In general the gaussian filters are anisotropic but when using the same sigma for x-, and y-direction they
 are isotropic.

:param grid:        the construction method.        enum {'discrete', 'continuous'}
:param order:       total order of the filter bank. integer
:param sigma_x:     sigma in x-direction.           float
:param sigma_y:     sigma in y-direction.           float
:param normalize:   normalization of the filters.   boolean

:return:            A rank 3 tensor containing the filters.
					The dimensions of the tensor denote [filter_number, y, x]
					The numbering follows: G, Gx, Gy, Gxx, Gxy, Gyy, Gxxx, Gxxy, Gyyx, Gyyy, Gxxxx,etc.
"""

@ops.RegisterShape("GaussianBasisFilters")
def _gaussian_basis_filters_shape(op):
	'''  
		shape inference function for GaussianBasisFilters
	'''
    # Get the attributes
	order = op.get_attr("order")

    # Compute number of filters
	N = (order + 1) * ((order + 1) - 1) / 2 + (order + 1)

    # We cannot infer the width and height of the filter because it's dynamic
	return [tf.TensorShape([tf.Dimension(None), tf.Dimension(None), tf.Dimension(N)])]

@ops.RegisterGradient("GaussianBasisFilters")
def _gaussian_basis_filters_grad(op, grad):
	''' 
		gradient function for GaussianBasisFilters
		which is then used for automatic differentation through the computation graph
	'''

	grid = op.get_attr("grid")
	order = op.get_attr("order")
	normalize = op.get_attr("normalize")

	# Load module
	_gaussian_basis_filters_backprop_sigma_module = load_library.load_op_library(os.path.join(resource_loader.get_data_files_path(), 'gaussian_basis_filters_backprop_sigma.so'))
	
	# Register operation
	sigma_x_grad, sigma_y_grad = _gaussian_basis_filters_backprop_sigma_module.gaussian_basis_filters_backprop_sigma(grid=grid, order=order, normalize=normalize, sigma_x=op.inputs[0], sigma_y=op.inputs[1], grad=grad)

	# List of two Tensors, since we have two sigmas
	return [sigma_x_grad, sigma_y_grad]


