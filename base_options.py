#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
# from util import util
import torch

class BaseOptions():
	def __init__(self):
		self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
		self.initialized = False

	def initialize(self):
		self.parser.add_argument(
			"--audio_dir", type=str, default='data/feats/vggish/', help="audio dir")
		self.parser.add_argument(
			"--video_dir", type=str, default='data/feats/res152/',
			help="video dir")
		self.parser.add_argument(
			"--st_dir", type=str, default='data/feats/r2plus1d_18/',
			help="video dir")
		self.parser.add_argument(
			"--label_train", type=str, default="data/AVVP_train.csv", help="weak train csv file")
		self.parser.add_argument(
			"--label_val", type=str, default="data/AVVP_val_pd.csv", help="weak val csv file")
		self.parser.add_argument(
			"--label_test", type=str, default="data/AVVP_test_pd.csv", help="weak test csv file")
		self.parser.add_argument('--batch_size', type=int, default=16, metavar='N',
							help='input batch size for training (default: 16)')
		self.parser.add_argument('--epochs', type=int, default=40, metavar='N',
							help='number of epochs to train (default: 60)')
		############# yb param ###########
		self.parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
							help='learning rate (default: 3e-4)')
		self.parser.add_argument('--lr_a', type=float, default=3e-4, metavar='LR',
							help='learning rate (default: 3e-4)')
		self.parser.add_argument('--lr_v', type=float, default=3e-4, metavar='LR',
							help='learning rate (default: 3e-4)')
		
		self.parser.add_argument('--occ_dim', type=int, default=64, metavar='LR',
							help='learning rate (default: 3e-4)')

		self.parser.add_argument('--init_epoch', type=int, default=5, metavar='LR',
							help='learning rate (default: 3e-4)')
		############# yb param ###########
		self.parser.add_argument(
			"--model", type=str, default='MMIL_Net', help="with model to use")
		self.parser.add_argument(
			"--mode", type=str, default='train', help="with mode to use")
		self.parser.add_argument('--seed', type=int, default=1, metavar='S',
							help='random seed (default: 1)')
		self.parser.add_argument('--log-interval', type=int, default=50, metavar='N',
							help='how many batches to wait before logging training status')
		self.parser.add_argument(
			"--model_save_dir", type=str, default='models/', help="model save dir")
		self.parser.add_argument(
			"--checkpoint", type=str, default='MMIL_Net',
			help="save model name")
		self.parser.add_argument(
			'--gpu', type=str, default='0,1', help='gpu device number')
		self.parser.add_argument(
			'--wandb', type=int, default='0', help='weight and bias setup')

		self.parser.add_argument(
			'--is_v_ori', type=int, default='0', help='original visual features')

		self.parser.add_argument(
			'--is_a_ori', type=int, default='0', help='original audio features')

		self.parser.add_argument(
			'--tsne', type=int, default='0', help='run tsne or not')
		self.parser.add_argument(
			'--early_stop', type=int, default='5', help='weight and bias setup')

		self.parser.add_argument(
			'--threshold', type=float, default=0.5, help='weight and bias setup')

		
		self.parser.add_argument(
			"--tmp", type=float, default=0.5,
			help="video dir")
		self.parser.add_argument(
			"--smooth", type=float, default=1,
			help="video dir")
		### yb param ##
		self.parser.add_argument(
			'--margin1', type=float, default=0.05, help='weight and bias setup')

		self.parser.add_argument(
			'--alpha', type=float, default=1, help='weight and bias setup')
		self.parser.add_argument(
			'--beta', type=float, default=1, help='weight and bias setup')
		self.parser.add_argument(
			'--delta', type=float, default=1, help='weight and bias setup')
		self.parser.add_argument(
			'--gamma', type=float, default=1, help='weight and bias setup')
		self.parser.add_argument(
			'--decay', type=float, default=0.1, help='decay rate')
		self.parser.add_argument(
			'--decay_epoch', type=float, default=10, help='decay rate')

		self.parser.add_argument(
			'--augment', type=int, default=0, help='data augment')

		self.parser.add_argument(
			'--vis_smoothing', type=float, default=1, help='vis label smoothing')

		self.parser.add_argument(
			'--audio_smoothing', type=float, default=1, help='audio label smoothing')


		self.parser.add_argument(
			'--before_vis_smoothing', type=float, default=1, help='vis label smoothing')

		self.parser.add_argument(
			'--before_audio_smoothing', type=float, default=1, help='audio label smoothing')


		self.parser.add_argument(
			'--aug_type', type=str, default='vision', help='weight and bias setup')

		self.parser.add_argument(
			'--pos_pool', type=str, default='max', help='weight and bias setup')
		self.parser.add_argument(
			'--neg_pool', type=str, default='mean', help='weight and bias setup')
		self.parser.add_argument(
			'--exp', type=int, default=0, help='weight and bias setup')
		
		self.parser.add_argument(
			'--ybloss', type=int, default=1, help='decay rate')


		### for transformer ###
		self.parser.add_argument(
			'--num_layer', type=int, default=1, help='num layer')
		self.parser.add_argument(
			'--num_head', type=int, default=1, help='num layer')
		self.parser.add_argument(
			'--prob_drop', type=float, default=0.1, help='drop out')
		self.parser.add_argument(
			'--prob_drop_occ', type=float, default=0.1, help='drop out')
		self.parser.add_argument(
			'--forward_dim', type=int, default=512, help='drop out')


		self.parser.add_argument(
			'--epoch_remove', type=int, default=1, help='weight and bias setup')
		#######################
		self.parser.add_argument(
			'--audio_enc', type=int, default= 0, help='weight and bias setup')

		self.parser.add_argument(	
			'--num_remove', type=int, default= 4, help='num of instances removing')

	def parse(self):
		if not self.initialized:
			self.initialize()
		self.opt = self.parser.parse_args()

		# str_ids = self.opt.gpu_ids.split(',')
		# self.opt.gpu_ids = []
		# for str_id in str_ids:
		# 	id = int(str_id)
		# 	if id >= 0:
		# 		self.opt.gpu_ids.append(id)

		# # set gpu ids
		# if len(self.opt.gpu_ids) > 0:
		# 	torch.cuda.set_device(self.opt.gpu_ids[0])


		#I should process the opt here, like gpu ids, etc.
		args = vars(self.opt)
		print('------------ Options -------------')
		for k, v in sorted(args.items()):
			print('%s: %s' % (str(k), str(v)))
		print('-------------- End ----------------')


		# save to the disk
		# expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
		# util.mkdirs(expr_dir)
		# file_name = os.path.join(expr_dir, 'opt.txt')
		# with open(file_name, 'wt') as opt_file:
		# 	opt_file.write('------------ Options -------------\n')
		# 	for k, v in sorted(args.items()):
		# 		opt_file.write('%s: %s\n' % (str(k), str(v)))
		# 	opt_file.write('-------------- End ----------------\n')
		return self.opt
