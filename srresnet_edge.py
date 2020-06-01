import tensorflow as tf


class SRresnetEdge:


	def __init__(self,weight_lamda=1, learning_rate=1e-4):
		""" Init class attr """
		self.learning_rate = learning_rate
		self.weight_lamda = weight_lamda
		

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
	    x = tf.nn.relu(x)
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


	def RecurrentBlock(self, x, k):
		f_k_in = x
		weights,biases = self.recurrent_weight(k)

		for i in range(k):
			f_k_mid = tf.nn.relu(tf.nn.conv2d(f_k_in, weights['w_in_%d'%(i)], strides=[1,1,1,1], padding='SAME') + biases['b_in_%d'%(i)])
			f_k = tf.nn.relu(tf.nn.conv2d(f_k_mid, weights['w_mid_%d'%(i)], strides=[1,1,1,1], padding='SAME') + biases['b_mid_%d'%(i)]) + f_k_in
			f_k_in = f_k
			print(f_k_in)
		return f_k_in

	def recurrent_weight(self, k):
		weights = {}
		biases = {}
		# with tf.variable_scope('sr_edge_net') as scope
		for i in range(k):
			weights.update({'w_in_%d'%(i):tf.Variable(tf.random_normal([3,3,64,64],stddev=1e-3),name='w_in_%d'%(i))})
			weights.update({'w_mid_%d'%(i):tf.Variable(tf.random_normal([3,3,64,64],stddev=1e-3),name='w_mid_%d'%(i))})
			biases.update({'b_in_%d'%(i):tf.Variable(tf.zeros([64]),name='b_in_%d'%(i))})
			biases.update({'b_mid_%d'%(i):tf.Variable(tf.zeros([64]),name='b_mid_%d'%(i))})

		return weights,biases

	def foward(self, x_in, x_edge, b_block=3):
		with tf.variable_scope('sr_edge_net') as scope:
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

			biases = {
				'b_in': tf.Variable(tf.zeros([64],name='b_in')),
				'b_edge': tf.Variable(tf.zeros([4],name='b_edge')),
				'b_rect': tf.Variable(tf.zeros([1],name='b_rect'))
			}

			# print(x_concate)
			x = tf.nn.conv2d(x_concate, weights['w_in'], strides=[1,1,1,1], padding='SAME', name='x_input') + biases['b_in']
			x = tf.nn.relu(x, name='x_input')
			skip = x
			x = self.RecurrentBlock(x,b_block)
			# for i in range(b_block):
			# 	x = self.ResidualBlock(x, 3, 64)
			print('____DEBUG____')
			print(x)
			"""
			f_output
			"""

			x_output = tf.nn.conv2d(x, weights['w_in_residual_out'], strides=[1,1,1,1], padding='SAME', name='w_in_residual_out')
			x_output = tf.nn.relu(x_output)

			# print(x_output)

			"""
			f_edge
			"""

			x_edge = tf.nn.conv2d(x_output, weights['w_edge'], strides=[1,1,1,1], padding='SAME', name='f_edge') + biases['b_edge']
			x_edge = tf.nn.relu(x_edge, name='f_edge')

			# print(x_output)
			# print(x_edge)
			
			"""
			f_rect
			"""
			x_rect = tf.concat([x_output,x_edge],axis=3, name='concate')
			# print(x_rect)

			x_H_hat = tf.nn.conv2d(x_rect, weights['w_rect'], strides=[1,1,1,1], padding='SAME', name='f_rect') + biases['b_rect']
			x_H_hat = tf.nn.relu(x_H_hat, name='f_rect')
			print(x_H_hat)

			# #Here add have some problem
			x_H = tf.math.add(x_H_hat, x_in, name='x_H_add')

			print(x_H)
			return x_H, x_H_hat


	def rect_loss(self, y_HR_hat, y_predict):
		with tf.variable_scope('sr_edge_net') as scope:
			return  (y_HR_hat - y_predict)


	def rect_loss(self, y_HR_hat, y_predict_HR):
		return  (tf.square(y_HR_hat - y_predict_HR))


	def edge_loss(self, y_edge_HR_hat, y_predict_edge_HR):
		return (tf.square(y_edge_HR_hat - y_predict_edge_HR))

	def total_loss(self, y_HR_hat, y_predict_HR, y_edge_HR_hat, y_predict_edge_HR):
		""" Not sure about joint loss  """
		return tf.reduce_mean(tf.square(y_HR_hat - y_predict_HR)+
			self.weight_lamda*(tf.square(tf.math.subtract(y_edge_HR_hat,y_predict_edge_HR))))

	# def optimizer(self, loss):
	# 	return tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

	def optimizer(self, loss):
		# tf.control_dependencies([discrim_train
		# update_ops needs to be here for batch normalization to work
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='sr_edge_net')
		with tf.control_dependencies(update_ops):
			return tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss, var_list=tf.get_collection(
				tf.GraphKeys.TRAINABLE_VARIABLES, scope='sr_edge_net'))