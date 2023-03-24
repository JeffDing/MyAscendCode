# -*- coding: utf-8 -*-
"""
Created on Thu May 30 14:32:05 2019

@author: clausmichele
"""

import argparse
from glob import glob
import sys
import tensorflow as tf
import os
from model_spatialCNN import denoiser
from utilis import *
import numpy as np

import moxing as mox

from npu_bridge.npu_init import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='epoch', type=int, default=1000, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help='# images in batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.0015, help='initial learning rate for adam')
parser.add_argument('--use_npu', dest='use_npu', type=int, default=1, help='npu flag, 1 for npu and 0 for CPU')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='/cache/VideNN/Spatial-CNN/ckpt', help='checkpoints are saved here')
parser.add_argument('--save_dir', dest='save_dir', default='/cache/VideNN/Spatial-CNN/data/denoised', help='denoised sample are saved here')
parser.add_argument('--data_url', dest='data_url', help='dataset are saved here')
parser.add_argument('--train_url', dest='train_url', help='train file are saved here')
args = parser.parse_args()

def sortKeyFunc(s):
	 return int(os.path.basename(s)[:-4])
	 
def denoiser_train(denoiser, lr):
	with load_data('/cache/VideNN/Spatial-CNN/data/train/img_clean_pats.npy') as data_:
		data = data_
	with load_data('/cache/VideNN/Spatial-CNN/data/train/img_noisy_pats.npy') as data_noisy_:
		data_noisy = data_noisy_

	noisy_eval_files = glob('/cache/VideNN/CBSD68/noisy15/*.png')
	noisy_eval_files = sorted(noisy_eval_files)
	eval_data_noisy = load_images(noisy_eval_files)
	eval_files = glob('/cache/VideNN/CBSD68/original_png/*.png')
	eval_files = sorted(eval_files)

	eval_data = load_images(eval_files)
	denoiser.train(data, data_noisy, eval_data[0:20], eval_data_noisy[0:20], batch_size=args.batch_size, ckpt_dir=args.ckpt_dir, epoch=args.epoch, lr=lr)
	mox.file.copy_parallel(args.ckpt_dir,args.train_url)


def denoiser_test(denoiser):
	noisy_eval_files = glob('/cache/VideNN/CBSD68/noisy15/*.png')
	noisy_eval_files = sorted(noisy_eval_files)
	eval_files = glob('/cache/VideNN/CBSD68/original_png/*.png')
	eval_files = sorted(eval_files)
	denoiser.test(noisy_eval_files, eval_files, ckpt_dir=args.ckpt_dir, save_dir=args.save_dir)

def denoiser_for_temp3_training(denoiser):
	noisy_eval_files = glob('/cache/VideNN/Temp3-CNN/data/train/noisy/*/*.png')
	noisy_eval_files = sorted(noisy_eval_files)
	eval_files = glob('/cache/VideNN/Temp3-CNN/data/train/original/*/*.png')
	eval_files = sorted(eval_files)
	denoiser.test(noisy_eval_files, eval_files, ckpt_dir=args.ckpt_dir, save_dir='/cache/VideNN/Temp3-CNN/data/train/denoised/')


def main(_):
	if not os.path.exists(args.ckpt_dir):
		os.makedirs(args.ckpt_dir)
	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)

	lr = args.lr * np.ones([args.epoch])
	lr[3:] = lr[0] / 10.0
	if args.use_npu:
		print("NPU\n")
		config = tf.ConfigProto()
		custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
		custom_op.name = "NpuOptimizer"
		config.graph_options.rewrite_options.remapping = RewriterConfig.OFF # 必须显式关闭remap
		config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF # 必须显式关闭
		custom_op.parameter_map["dynamic_input"].b = True
		custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
		custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes("lazy_recompile")
		with tf.Session(config=config) as sess:
			model = denoiser(sess)
			if args.phase == 'train':
				denoiser_train(model, lr=lr)
			elif args.phase == 'test':
				denoiser_test(model)
			elif args.phase == 'test_temp':
				denoiser_for_temp3_training(model)
			else:
				print('[!] Unknown phase')
				exit(0)
	else:
		print("CPU\n")
		with tf.device('/cpu:0'):
			with tf.Session() as sess:
				model = denoiser(sess)
				if args.phase == 'train':
					denoiser_train(model, lr=lr)
				elif args.phase == 'test':
					denoiser_test(model)
				elif args.phase == 'test_temp':
					denoiser_for_temp3_training(model)
				else:
					print('[!] Unknown phase')
					exit(0)


if __name__ == '__main__':
	mox.file.copy_parallel(args.data_url,'/cache/VideNN')
	tf.app.run()

