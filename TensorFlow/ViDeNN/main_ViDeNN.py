# -*- coding: utf-8 -*-
"""
@author: clausmichele
"""

import argparse
from glob import glob
import tensorflow as tf
import os

from model_ViDeNN import ViDeNN

from npu_bridge.npu_init import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('--use_npu', dest='use_npu', type=int, default=1, help='npu flag, 1 for nPU and 0 for CPU')
parser.add_argument('--save_dir', dest='save_dir', default='./data/denoised', help='denoised sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./data', help='directory of noisy frames')
parser.add_argument('--img_format', dest='img_format', default='png', help='denoised sample are saved here')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default=None, help='path of ViDeNN checkpoint')
args = parser.parse_args()

def ViDeNNDenoise(ViDeNN):
	eval_files_noisy = glob(args.test_dir + "/noisy/*." + args.img_format)
	eval_files_noisy = sorted(eval_files_noisy)
	eval_files = glob(args.test_dir + "/original/*." + args.img_format)
	print_psnr = True
	if eval_files == []:
		eval_files = eval_files_noisy
		print_psnr = False
		print("[*] No original frames found, not printing PSNR values...")
	eval_files = sorted(eval_files)
	ViDeNN.denoise(eval_files, eval_files_noisy, print_psnr, args.ckpt_dir, args.save_dir)

def main(_):
	if not os.path.exists(args.test_dir):
		os.makedirs(args.test_dir)
	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)
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
			model = ViDeNN(sess)
			ViDeNNDenoise(model)
	else:
		print("CPU\n")
		with tf.device('/cpu:0'):
			with tf.Session() as sess:
				model = ViDeNN(sess)
				ViDeNNDenoise(model)


if __name__ == '__main__':
	tf.app.run()

