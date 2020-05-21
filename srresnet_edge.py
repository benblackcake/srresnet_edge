import tensorflow as tf


class SRresnetEdge:


	def __init__(self):
		""" Init class attr """
		self.lamda = None
		pass

	def ResidualBlock(self, x, kernel_size, filter_size):
	    """Residual block a la ResNet"""

	    weights = {
	            'w1': tf.Variable(tf.random_normal([kernel_size, kernel_size, filter_size, filter_size], stddev=1e-3), name='w1'),
	            'w2': tf.Variable(tf.random_normal([kernel_size, kernel_size, filter_size, filter_size], stddev=1e-3), name='w2'),
	            # 'w3': tf.Variable(tf.random_normal([3, 3, 32, self.c_dim], stddev=1e-3), name='w3')
	        }

	    skip = x
	    x = tf.nn.conv2d(x, weights['w1'], strides=[1,1,1,1], padding='SAME')
	    x = tf.layers.batch_normalization(x)
	    x = tf.nn.relu(x)
	    x = tf.nn.conv2d(x, weights['w2'], strides=[1,1,1,1], padding='SAME')
	    x = tf.layers.batch_normalization(x)
	    x = x + skip
	    return x

	def Upsample2xBlock(self, x, kernel_size, filter_size):
		weights = {
		    'w1': tf.Variable(tf.random_normal([kernel_size, kernel_size, 64, filter_size], stddev=1e-3), name='w1'),
		}
		"""Upsample 2x via SubpixelConv"""
		print('init',x)
		x = tf.nn.conv2d(x, weights['w1'], strides=[1,1,1,1], padding='SAME')
		print('before',x)
		x = tf.depth_to_space(x, 2)
		print('after',x)
		
		x = tf.nn.relu(x)
		return x


	def foward(self, x_in, x_edge, b_block=8):

		# y_concate = x
		print(x_in)
		x_concate = tf.concat([x_in, x_edge],axis=3, name='x_input_concate')
		weights ={
			'w_in': tf.Variable(tf.random_normal([3,3,4,64], stddev=1e-3), name='w_in'),
			'w_in_residual_out': tf.Variable(tf.random_normal([3,3,64,64], stddev=1e-3), name='w_in_residual_out'),
			# 'w_in_out': tf.Variable(tf.random_normal([3,3,256,3], stddev=1e-3), name='w_in_residual_out'),
			# 'w_out': tf.Variable(tf.random_normal([3,3,64,64], stddev=1e-3), name='w_out'),
			'w_edge': tf.Variable(tf.random_normal([3,3,64,4], stddev=1e-3), name='w_edge'),
			'w_rect': tf.Variable(tf.random_normal([3,3,68,1], stddev=1e-3), name='w_rect')
			# 'w_H': tf.Variable(tf.random_normal([3,3,68,1], stddev=1e-3), name='w_H')

		}
		print(x_concate)
		x = tf.nn.conv2d(x_concate, weights['w_in'], strides=[1,1,1,1], padding='SAME', name='x_input')
		x = tf.nn.relu(x, name='x_input')
		skip = x
		for i in range(b_block):
			x = self.ResidualBlock(x, 3, 64)

		"""
		f_output
		"""

		x = tf.nn.conv2d(x, weights['w_in_residual_out'], strides=[1,1,1,1], padding='SAME', name='w_in_residual_out')
		x = tf.layers.batch_normalization(x)

		x_output = tf.math.add(x, skip, name='f_output')
		# for i in range(1):
		# 	x = Upsample2xBlock(x, kernel_size=3, filter_size=256)

		print(x_output)

		# """
		# f_edge
		# """
		# x_output = tf.nn.conv2d(x_output, weights['w_out'], strides=[1,1,1,1], padding='SAME', name='f_out')
		# x_output =  tf.nn.relu(x_output, name='x_output')
		x_edge = tf.nn.conv2d(x_output, weights['w_edge'], strides=[1,1,1,1], padding='SAME', name='f_edge')
		x_edge = tf.nn.relu(x_edge, name='f_edge')

		# print(x_output)
		print(x_edge)
		
		# """
		# f_rect
		# """
		x_rect = tf.concat([x_output,x_edge],axis=3, name='concate')
		print(x_rect)

		x_H_hat = tf.nn.conv2d(x_rect, weights['w_rect'], strides=[1,1,1,1], padding='SAME', name='f_rect')
		x_H_hat = tf.nn.relu(x_H_hat, name='f_rect')
		print(x_H_hat)

		# #Here add have some problem
		x_H = tf.math.add(x_H_hat, x_in, name='x_H_add')

		# # print(x_concate)
		# print(x_H)
		print(x_H)
		return x_H, x_edge


	def rect_loss(self, y_HR_hat, y_predict):
		return tf.square(y_HR_hat - y_predict)


	def edge_loss(self, y_edge_HR_hat, y_predict):
		return tf.square(y_edge_HR_hat - y_predict)

	def total_loss(self, rect_loss, edge_loss):
		""" Not sure about joint loss  """
		return  tf.reduce_mean(rect_loss) + tf.reduce_mean((1 * edge_loss))

	def optimizer(self, loss):
		return tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)