
# ------------------------------------------------------------------------------------------
# The pytorch demo code for detecting the object in a specific image (fpn specific version)
# Licensed under The MIT License [see LICENSE for details]
# Written by Jianwei Yang, modified by Zongxian Li, based on code from faster R-CNN
# ------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.utils.blob import im_list_to_blob
import os
import sys
import numpy as np
np.set_printoptions(suppress=True)
import argparse
import pprint
import pdb
import time
import cv2
import pickle as cPickle
import torch
from numpy.core.multiarray import ndarray
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
#from scipy.misc import imread
from imageio import imread
from roi_data_layer.roidb_v1 import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.soft_nms.nms import soft_nms as nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import vis_detections
# from model.fpn.fpn_cascade import _FPN
from model.fpn.resnet_IN import resnet
from model.fpn.normalization import Regularization
from astropy.io import fits
import pdb


def parse_args():
	"""
	Parse input arguments
	"""
	parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
	parser.add_argument('--webcam_num', dest='webcam_num',
	                    help='webcam ID number',
	                    default=-1, type=int)
	parser.add_argument('--dataset', dest='dataset',
	                    help='training dataset',
	                    default='star_detect', type=str)
	parser.add_argument('--cfg', dest='cfg_file',
	                    help='optional config file',
	                    default='cfgs/res50.yml', type=str)
	parser.add_argument('--net', dest='net',
	                    help='vgg16, res50, res101, res152',
	                    default='res50', type=str)
	parser.add_argument('--set', dest='set_cfgs',
	                    help='set config keys', default=None,
	                    nargs=argparse.REMAINDER)
	parser.add_argument('--load_dir', dest='load_dir',
	                    help='directory to load models', default="/home/lab30202/sdb/liuqiang/2020-2-11-star_detection_release/detection_release/fpn_v1/model_save",
	                    nargs=argparse.REMAINDER,type = str)
	parser.add_argument('--cuda', dest='cuda',
	                    help='whether use CUDA', default=1,
	                    action='store_true')
	parser.add_argument('--image_dir', dest='image_dir',
	                    help='directory to load images for demo',
	                    default="/home/lab30202/sdb/liuqiang/2020-2-11-star_detection_release/detection_release/fpn_v1/validation")
	parser.add_argument('--cag', dest='class_agnostic',
	                    help='whether perform class_agnostic bbox regression',
	                    action='store_true')
	parser.add_argument('--parallel_type', dest='parallel_type',
	                    help='which part of model to parallel, 0: all, 1: model before roi pooling',
	                    default=0, type=int)
	parser.add_argument('--checksession', dest='checksession',
	                    help='checksession to load model',
	                    default=100, type=int)
	parser.add_argument('--checkepoch', dest='checkepoch',
	                    help='checkepoch to load network',
	                    default=4, type=int)
	parser.add_argument('--checkpoint', dest='checkpoint',
	                    help='checkpoint to load network',
	                    default=2223, type=int)
	parser.add_argument('--bs', dest='batch_size',
	                    help='batch_size',
	                    default=1, type=int)
	parser.add_argument('--lr', dest='lr',
	                    help='starting learning rate',
	                    default=0.0005, type=float)
	parser.add_argument('--vis', dest='vis',
	                    help='visualization mode',
	                    default= True,
	                    action='store_true')
	args = parser.parse_args()
	return args


def _get_image_blob(im):
	"""Converts an image into a network input.
	Arguments:
	  im (ndarray): a color image in BGR order
	Returns:
	  blob (ndarray): a data blob holding an image pyramid
	  im_scale_factors (list): list of image scales (relative to im) used
		in the image pyramid
	"""
	im_orig = im.astype(np.float32, copy=True)
	im_shape = im_orig.shape
	im_size_min = np.min(im_shape[0:2])
	im_size_max = np.max(im_shape[0:2])

	processed_ims = []
	im_scale_factors = []

	for target_size in cfg.TEST.SCALES:
		im_scale = float(target_size) / float(im_size_min)
		# Prevent the biggest axis from being more than MAX_SIZE
		if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
			im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
		im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
		                interpolation=cv2.INTER_LINEAR)
		im_scale_factors.append(im_scale)
		processed_ims.append(im)

	# Create a blob to hold the input images
	blob = im_list_to_blob(processed_ims)

	return blob, np.array(im_scale_factors)


if __name__ == '__main__':

	os.environ["CUDA_VISIBLE_DEVICES"] = "1"
	print("GPU {} will be used\n".format("1"))

	args = parse_args()

	lr = args.lr
	momentum = cfg.TRAIN.MOMENTUM
	weight_decay = cfg.TRAIN.WEIGHT_DECAY

	print('Called with args:')
	print(args)
	cfg.USE_GPU_NMS = args.cuda
	if args.cfg_file is not None:
		cfg_from_file(args.cfg_file)
	if args.set_cfgs is not None:
		cfg_from_list(args.set_cfgs)

	print('Using config:')
	pprint.pprint(cfg)
	np.random.seed(cfg.RNG_SEED)

	input_dir = os.path.join(args.load_dir[0], args.net,args.dataset)
	if not os.path.exists(input_dir):
		raise Exception('There is no input directory for loading network from ' + input_dir)
	load_name = os.path.join(input_dir,
	                         'fpn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
	print(load_name)

	pascal_classes = np.asarray(['__background__','star',"galaxy"])
	# initilize the network here.
	if args.net == 'res101':
		fpn = resnet(pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
	elif args.net == 'res50':
		fpn = resnet(pascal_classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
	elif args.net == 'res152':
		fpn = resnet(pascal_classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
	else:
		print("network is not defined")
		pdb.set_trace()

	fpn.create_architecture()
	fpn.cuda()
	print("load checkpoint %s" % (load_name))
	if args.cuda > 0:
		checkpoint = torch.load(load_name)
	fpn.load_state_dict(checkpoint['model'])
	if 'pooling_mode' in checkpoint.keys():
		cfg.POOLING_MODE = checkpoint['pooling_mode']

	print('load model successfully!')

	im_data = torch.FloatTensor(1)
	im_info = torch.FloatTensor(1)
	num_boxes = torch.LongTensor(1)
	gt_boxes = torch.FloatTensor(1)

	# ship to cuda
	if args.cuda > 0:
		im_data = im_data.cuda()
		im_info = im_info.cuda()
		num_boxes = num_boxes.cuda()
		gt_boxes = gt_boxes.cuda()

	# make variable
	im_data = Variable(im_data)
	im_info = Variable(im_info)
	num_boxes = Variable(num_boxes)
	gt_boxes = Variable(gt_boxes)

	if args.cuda > 0:
		cfg.CUDA = True

	if args.cuda > 0:
		fpn.cuda()

	fpn.eval()

	start = time.time()
	max_per_image = 2500
	thresh = 0.05
	vis = True
	webcam_num = args.webcam_num
	# Set up webcam or get image directories

	if webcam_num >= 0:
		cap = cv2.VideoCapture(webcam_num)    
		num_images = 0
	else:                                      
		import glob
		fits_list = os.path.join(args.image_dir,'*.fits')
		imglist = glob.glob(fits_list)

		num_images = len(imglist)

	print('Loaded Photo: {} images.'.format(num_images))

	while (num_images >= 0):
		box_list = list()
		total_tic = time.time()
		if webcam_num == -1:
			num_images -= 1

		# Get image from the webcam
		if webcam_num >= 0:
			if not cap.isOpened():
				raise RuntimeError("Webcam could not open. Please check connection.")
			ret, frame = cap.read()
			im_in = np.array(frame)
		# Load the demo image
		else:
			im_file = os.path.join(args.image_dir, imglist[num_images])

			im = fits.open(im_file,ignore_missing_end = True)[0].data
			# ### use log transpoze
			# im = np.log(1 + np.abs(im))
			max_value = np.max(im)
			min_value = np.min(im)
			mean_value = np.mean(im)
			var_value = np.var(im)
			im_in = (im - mean_value) / (max_value - min_value)
			# im_in = (im - mean_value)/var_value


		if len(im_in.shape) == 2:
			im_in = im_in[:, :, np.newaxis]
			im_in = np.concatenate((im_in, im_in, im_in), axis=2)
		im = im_in[:, :, ::-1]

		blobs, im_scales = _get_image_blob(im)  
		assert len(im_scales) == 1, "Only single-image batch implemented"
		im_blob = blobs
		im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

		im_data_pt = torch.from_numpy(im_blob)
		im_data_pt = im_data_pt.permute(0, 3, 1, 2)
		im_info_pt = torch.from_numpy(im_info_np)

		im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
		# print(im_data.shape)
		im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
		# print(im_info.shape)
		gt_boxes.resize_(1, 5).zero_()
		# print(gt_boxes.shape)
		num_boxes.resize_(1).zero_()
		# print(num_boxes.shape)

		# pdb.set_trace()
		det_tic = time.time()

		rois, cls_prob, bbox_pred, \
		_, _, _, _, _ = fpn(im_data, im_info, gt_boxes, num_boxes)

		scores = cls_prob.data
		boxes = rois.data[:, :, 1:5]

		if cfg.TEST.BBOX_REG:
			# Apply bounding-box regression deltas
			box_deltas = bbox_pred.data
			if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
				# Optionally normalize targets by a precomputed mean and stdev
				if args.class_agnostic:
					if args.cuda > 0:
						box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
						             + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
					else:
						box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
						             + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

					box_deltas = box_deltas.view(1, -1, 4)
				else:
					if args.cuda > 0:
						box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
						             + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
					else:
						box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
						             + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
					box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

			pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
			pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
		else:
			pred_boxes = np.tile(boxes, (1, scores.shape[1]))

		pred_boxes /= im_scales[0]

		scores = scores.squeeze()
		pred_boxes = pred_boxes.squeeze()
		det_toc = time.time()
		detect_time = det_toc - det_tic
		misc_tic = time.time()

		if vis:
			jpg_name = os.path.join(args.image_dir,im_file.split("/")[-1].split(".fits")[0] + ".jpg")
			im_jpg = cv2.imread(jpg_name)
			im_jpg = cv2.flip(im_jpg,0)
			im2show = np.copy(im_jpg)
		for j in range(1, len(pascal_classes)):
			inds = torch.nonzero(scores[:, j] > thresh).view(-1)
			# if there is det
			if inds.numel() > 0:
				cls_scores = scores[:, j][inds]
				_, order = torch.sort(cls_scores, 0, True)
				if args.class_agnostic:
					cls_boxes = pred_boxes[inds, :]
				else:
					cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

				cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
				# cls_dets = torch.cat((cls_boxes, cls_scores), 1)
				cls_dets = cls_dets[order]
				keep = nms(cls_dets.cpu())  ##　use soft_nms
				cls_dets = keep

				if pascal_classes[j] == "star":
					class_name_index = 100
					class_name_column = [class_name_index]*cls_dets.shape[0]
					class_name = np.array(class_name_column).reshape(len(class_name_column),1)
					cls_dets = np.concatenate([cls_dets,class_name],axis=1)
				else:
					class_name_index = 200
					class_name_column = [class_name_index]*cls_dets.shape[0]
					class_name = np.array(class_name_column).reshape(len(class_name_column),1)
					cls_dets = np.concatenate([cls_dets,class_name],axis=1)

				box_list.append(cls_dets)
				if vis:
					im2show,save_mat = vis_detections(im2show, pascal_classes[j], cls_dets, 0.5)



		result_path = os.path.join(args.image_dir, imglist[num_images][:-4].rstrip(".") + "_det.txt")
		box_np = np.concatenate(box_list, axis=0)
		np.savetxt(result_path,box_np,fmt="%.8f")
		if vis and webcam_num == -1:
			result_path = os.path.join(args.image_dir, imglist[num_images][:-4] + "_det.jpg")
			cv2.imwrite(result_path, im2show,[int(cv2.IMWRITE_JPEG_QUALITY), 100])

		misc_toc = time.time()
		nms_time = misc_toc - misc_tic

print("completed!!!")

