import tensorflow as tf
import numpy as np

class GaussianBasisFiltersGradTest(tf.test.TestCase):
	
	def testGradientChecking(self):
		x = tf.Variable(1.0)
		y = tf.user_ops.gaussian_basis_filters(grid="continuous", order=0, sigma_x=x, sigma_y=x, normalize=True)
		
		# Compute MSE
		y_target = tf.ones_like(y)
		J = tf.reduce_mean(tf.square(tf.subtract(y_target, y))) 
		
		with tf.Session() as sess:
			
			tf.initialize_all_variables().run()
			
			error =  tf.test.compute_gradient_error(x, (), J, ())
			# print(error)
			self.assertLess(error, 1e-4)
			
			analytical, numerical = tf.test.compute_gradient(x, (), J, (), x_init_value=np.asarray(1.23))
			# print(analytical, numerical)
			self.assertAlmostEqual(analytical, numerical, places=4)
			
	def testGradient(self):
		sigma_x = tf.Variable(1.0)
		sigma_y = tf.Variable(1.0)
		y = tf.user_ops.gaussian_basis_filters(grid="discrete", order=9, sigma_x=sigma_x, sigma_y=sigma_y, normalize=True)
		
		error_signals = tf.ones_like(y)
		
		with tf.Session() as sess:
		
			tf.initialize_all_variables().run()
		
			'''
			grad = sess.run(tf.gradients(y, x, error_signals))
			print(grad)
			self.assertAlmostEqual(grad[0], -0.3273, places=4)
			'''
			
			grad = sess.run(tf.gradients(y, [sigma_x, sigma_y], error_signals))
			print(grad)
			self.assertAlmostEqual((grad[0]+grad[1]), -0.3273, places=4)
			
			#print(x.eval())
			#print(grad)
			
			#print(tf.shape(x))
			#print(tf.shape(grad[0]))

if __name__ == "__main__":
	# Run all unit tests
	tf.test.main()
  
