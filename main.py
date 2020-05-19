

from srresnet_edge import *
import tensorflow as tf
import argparse
from utils import *
import numpy as np
import os
import sys
import cv2
from tqdm import tqdm,trange
from benchmark import Benchmark

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch-size', type=int, default=16, help='Mini-batch size.')
	parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate for Adam.')
    parser.add_argument('--train-dir', type=str, help='Directory containing training images')
	parser.add_argument('--image-size', type=int, default=96, help='Size of random crops used for training samples.')
	parser.add_argument('--epoch', type=int, default='100', help='How many iterations ')
	parser.add_argument('--log-freq', type=int, default=10000, help='How many training iterations between validation/checkpoints.')
	parser.add_argument('--is-val', action='store_true', help='True for evaluate image')
	parser.add_argument('--gpu', type=str, default='0', help='Which GPU to use')

	args = parser.parse_args()
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

	args = parser.parse_args()

	"""
	Testing Variable 
	"""
	# t_path = "D://code_IMP//python//tmp//np_test//output"
	t_path = str(args.train_dir)
	
	x = tf.Variable(tf.random_normal([128,33, 33, 3], stddev=1e-3), name='x_LR')
	x_edge = tf.Variable(tf.random_normal([128,33, 33, 1], stddev=1e-3), name='x_LR_edge')

	"""
	Image and Edge map placeholder
	"""
	hr_ = tf.placeholder(tf.float32, [None, None, None, 3], name='HR_image')
	lr_ = tf.placeholder(tf.float32, [None, None, None, 3], name='LR_image')

	hr_edge = tf.placeholder(tf.float32, [None, None, None, 1], name='HR_edge') 
	lr_edge = tf.placeholder(tf.float32, [None, None, None, 1], name='LR_edge') 

	# x_concatenation = tf.concat([x, x_edge],axis=3, name='x_input_concate')
	# conv_input = tf.nn.conv2d(x, weight_input, strides=[1,1,1,1], padding='SAME')

	""" DEBUG placeholder parmaters """

	print(hr_)
	print(lr_ )
	print(hr_edge)
	print(lr_edge)


	x_H, x_edge = foward(lr_, lr_edge)

	edgeLoss = edge_loss(hr_edge, x_edge)
	rectLoss = rect_loss(hr_, x_H)
	totalLoss = total_loss(edgeLoss, rectLoss)

	totalLoss_opt = optimizer(totalLoss)


	benchmarks = [
		Benchmark('Benchmarks/Set5', name='Set5'),
		Benchmark('Benchmarks/Set14', name='Set14'),
		Benchmark('Benchmarks/BSD100', name='BSD100')
	]


	"""Train session"""
	with tf.Session() as sess:
		sess.run(tf.local_variables_initializer())
		sess.run(tf.global_variables_initializer())
		# print(get_batch_folder_list(t_path))
		# print(iter(get_batch_folder_list(t_path)))
		saver = tf.train.Saver()

		iterator = 0
		load(sess,saver,'checkpoint')


		""" To validate Benchmarks"""
		if args.is_val:
			for benchmark in benchmarks:
				psnr, ssim, _, _ = benchmark.eval(sess, x_H, x_edge, iterator)
				print(' [%s] PSNR: %.2f, SSIM: %.4f' % (benchmark.name, psnr, ssim), end='')

		""" To Training """
		else:
			for epoch in range(args.epoch):
				file_pathes = get_batch_folder_list(t_path)
				t = tqdm(range(len(file_pathes)), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', desc='Iterations')
				#bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'

				for file_path_indx in t:
					image_list = get_batch_image(file_pathes[file_path_indx])

					hr_img_batch = np.asarray(image_list)
					hr_edges_map = cany_oper_batch(hr_img_batch)
					hr_edges_map = np.expand_dims(hr_edges_map,axis=-1)

					lr_img_batch = downsample_batch(hr_img_batch,3)
					lr_edges_map = cany_oper_batch(lr_img_batch.astype(np.uint8))
					lr_edges_map = np.expand_dims(lr_edges_map,axis=-1)
			
					hr_img_batch = normalize_batch_img(hr_img_batch)
					hr_edges_map = normalize_batch_img(hr_edges_map)
					lr_img_batch = normalize_batch_img(lr_img_batch)
					lr_edges_map = normalize_batch_img(lr_edges_map)

					# for canyed in hr_edges_map:
					# 	cv2.imshow("cany edges",canyed)
					# 	cv2.waitKey()
					# print(hr_img_batch.shape)
					# print(hr_edges_map.shape)
					# print(lr_img_batch.shape)
					# print(lr_edges_map.shape)

					_,err = sess.run([totalLoss_opt,totalLoss], feed_dict={hr_:hr_img_batch,hr_edge:hr_edges_map,
													   lr_:lr_img_batch,lr_edge:lr_edges_map})

					t.set_description("[Iters: %s][Error: %.4f]" %(iterator,err))

					if iterator%args.log_freq == 0:
						save(sess,saver,'checkpoint',iterator)
					iterator+=1

				# print(err)

if __name__=='__main__':

	main()