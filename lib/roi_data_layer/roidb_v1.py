"""Transform a roidb into a trainable roidb by adding a bunch of metadata."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append('..')
from  astropy.io import fits

import numpy as np
from model.utils.config import cfg
import datasets
from datasets.factory import get_imdb
import pdb
def prepare_roidb(imdb):
	"""Enrich the imdb's roidb by adding some derived quantities that
	are useful for training. This function precomputes the maximum
	overlap, taken over ground-truth boxes, between each ROI and
	each ground-truth box. The class with maximum overlap is also
	recorded.
	"""

	roidb = imdb.roidb
	# 对所有的iamge（包含数据增强部分）进行迭代
	for i in range(len(imdb.image_index)):
		# image信息记录图像全路径，width、heigth为图片宽和高
		roidb[i]['img_id'] = imdb.image_id_at(i)
		roidb[i]['image'] = imdb.image_path_at(i)
		if not (imdb.name.startswith('coco')):
			from astropy.io import fits
			im = fits.open(imdb.image_path_at(i), ignore_missing_end=True)[0].data
			roidb[i]['width'] = im.shape[1]
			roidb[i]['height'] = im.shape[0]
		# need gt_overlaps as a dense array for argmax
		# roidb[i]['gt_overlaps']为压缩后的one-hot矩阵，toarray()就为解压缩，复原了one_hot矩阵
		gt_overlaps = roidb[i]['gt_overlaps'].toarray()
		# max overlap with gt over classes (columns)
		# 取出最大值
		max_overlaps = gt_overlaps.max(axis=1)
		# gt class that had the max overlap
		max_classes = gt_overlaps.argmax(axis=1)
		# 在roidb列表中的图片信息dict中添加两个信息
		roidb[i]['max_classes'] = max_classes
		roidb[i]['max_overlaps'] = max_overlaps
		# sanity checks
		# max overlap of 0 => class should be zero (background)
		zero_inds = np.where(max_overlaps == 0)[0]
		assert all(max_classes[zero_inds] == 0)
		# max overlap > 0 => class should not be zero (must be a fg class)
		nonzero_inds = np.where(max_overlaps > 0)[0]
		assert all(max_classes[nonzero_inds] != 0)



def rank_roidb_ratio(roidb):
	ratio_list = []
	for i in range(len(roidb)):
		ratio = 1
		ratio_list.append(ratio)
		roidb[i]['need_crop'] = 0
	ratio_list = np.array(ratio_list)
	ratio_index = np.argsort(ratio_list)
	return ratio_list[ratio_index], ratio_index


def filter_roidb(roidb):
	# filter the image without bounding box.
	print('before filtering, there are %d images...' % (len(roidb)))
	i = 0
	while i < len(roidb):
		if len(roidb[i]['boxes']) == 0:
			del roidb[i]
			i -= 1
		i += 1

	print('after filtering, there are %d images...' % (len(roidb)))
	return roidb


def combined_roidb(imdb_names, use_filpped ,training=True):
	"""
	Combine multiple roidbs
	"""

	def get_training_roidb(imdb):
		"""Returns a roidb (Region of Interest database) for use in training."""
		if use_filpped:     ###　　whether filp the image
			print('Appending horizontally-flipped training examples...')
			imdb.append_flipped_images()
			print('done')

		print('Preparing training data...')

		prepare_roidb(imdb)
		print('done')

		return imdb.roidb

	def get_roidb(imdb_name):
		imdb = get_imdb(imdb_name)
		print('Loaded dataset `{:s}` for training'.format(imdb.name))
		imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
		print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
		roidb = get_training_roidb(imdb)
		return roidb

	roidbs = [get_roidb(s) for s in imdb_names.split('+')]
	roidb = roidbs[0]

	if len(roidbs) > 1:
		for r in roidbs[1:]:
			roidb.extend(r)
		tmp = get_imdb(imdb_names.split('+')[1])
		imdb = datasets.imdb_v1.imdb(imdb_names, tmp.classes)
	else:
		imdb = get_imdb(imdb_names)

	if training:
		roidb = filter_roidb(roidb)

	ratio_list, ratio_index = rank_roidb_ratio(roidb)

	return imdb, roidb,ratio_list, ratio_index


if __name__ == '__main__':
	imdb_names = 'traindata300-300'
	usefilped = False
	imdb, roidb,ratio_list, ratio_index  = combined_roidb(imdb_names,usefilped)
	from IPython import embed;
	embed()


