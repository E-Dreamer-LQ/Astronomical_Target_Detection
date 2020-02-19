import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, gradcheck
from torch.autograd.gradcheck import gradgradcheck
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
import torchvision.utils as vutils
from model.utils.config import cfg
from model.rpn.rpn_fpn import _RPN_FPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer import _ProposalTargetLayer
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta
from model.utils.net_utils import FocalLoss
from model.fpn.Vgg_mine import Cnn
import time
import pdb


class _FPN_emsemble(nn.Module):
	""" FPN """

	def __init__(self, classes, class_agnostic):
		super(_FPN_emsemble, self).__init__()
		self.classes = classes
		self.n_classes = len(classes)
		self.class_agnostic = class_agnostic
		# loss
		self.RCNN_loss_cls = 0
		self.RCNN_loss_bbox = 0

		self.maxpool2d = nn.MaxPool2d(1, stride=2)
		# define rpn
		self.RCNN_rpn = _RPN_FPN(self.dout_base_model)
		self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

		# NOTE: the original paper used pool_size = 7 for cls branch, and 14 for mask branch, to save the
		# computation time, we first use 14 as the pool_size, and then do stride=2 pooling for cls branch.
		self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0)  ## ori:1/16.0
		self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0)  ## ori:1/16.0
		self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
		self.RCNN_roi_crop = _RoICrop()

	def _init_weights(self):
		def normal_init(m, mean, stddev, truncated=False):
			"""
			weight initalizer: truncated normal and random normal.
			"""
			# x is a parameter
			if truncated:
				m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
			else:
				m.weight.data.normal_(mean, stddev)
				m.bias.data.zero_()

		# custom weights initialization called on netG and netD
		def weights_init(m, mean, stddev, truncated=False):
			classname = m.__class__.__name__
			if classname.find('Conv') != -1:
				m.weight.data.normal_(0.0, 0.02)
				m.bias.data.fill_(0)
			elif classname.find('BatchNorm') != -1:
				m.weight.data.normal_(1.0, 0.02)
				m.bias.data.fill_(0)

		normal_init(self.RCNN_toplayer, 0, 0.01, cfg.TRAIN.TRUNCATED)
		normal_init(self.RCNN_smooth1, 0, 0.01, cfg.TRAIN.TRUNCATED)
		normal_init(self.RCNN_smooth2, 0, 0.01, cfg.TRAIN.TRUNCATED)
		normal_init(self.RCNN_smooth3, 0, 0.01, cfg.TRAIN.TRUNCATED)
		normal_init(self.RCNN_latlayer1, 0, 0.01, cfg.TRAIN.TRUNCATED)
		normal_init(self.RCNN_latlayer2, 0, 0.01, cfg.TRAIN.TRUNCATED)
		normal_init(self.RCNN_latlayer3, 0, 0.01, cfg.TRAIN.TRUNCATED)
		normal_init(self.RCNN_toplayer_det, 0, 0.01, cfg.TRAIN.TRUNCATED)
		normal_init(self.RCNN_smooth1_det, 0, 0.01, cfg.TRAIN.TRUNCATED)
		normal_init(self.RCNN_smooth2_det, 0, 0.01, cfg.TRAIN.TRUNCATED)
		# normal_init(self.RCNN_smooth3, 0, 0.01, cfg.TRAIN.TRUNCATED)
		normal_init(self.RCNN_latlayer1_det, 0, 0.01, cfg.TRAIN.TRUNCATED)
		normal_init(self.RCNN_latlayer2_det, 0, 0.01, cfg.TRAIN.TRUNCATED)
		normal_init(self.RCNN_latlayer3_det, 0, 0.01, cfg.TRAIN.TRUNCATED)
		normal_init(self.RCNN_latlayer4_det, 0, 0.01, cfg.TRAIN.TRUNCATED)

		normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
		normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
		normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
		normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
		normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)
		weights_init(self.RCNN_top, 0, 0.01, cfg.TRAIN.TRUNCATED)


		normal_init(self.RCNN_cls_score_det, 0, 0.01, cfg.TRAIN.TRUNCATED)
		normal_init(self.RCNN_bbox_pred_det, 0, 0.001, cfg.TRAIN.TRUNCATED)
		weights_init(self.RCNN_top_det, 0, 0.01, cfg.TRAIN.TRUNCATED)

	def create_architecture(self):
		self._init_modules()
		self._init_weights()

	def _upsample_add(self, x, y):
		'''Upsample and add two feature maps.
		Args:
		  x: (Variable) top feature map to be upsampled.
		  y: (Variable) lateral feature map.
		Returns:
		  (Variable) added feature map.
		Note in PyTorch, when input size is odd, the upsampled feature map
		with `F.upsample(..., scale_factor=2, mode='nearest')`
		maybe not equal to the lateral feature map size.
		e.g.
		original input size: [N,_,15,15] ->
		conv2d feature map size: [N,_,8,8] ->
		upsampled feature map size: [N,_,16,16]
		So we choose bilinear upsample which supports arbitrary output sizes.
		'''
		_, _, H, W = y.size()
		return F.upsample(x, size=(H, W), mode='bilinear') + y

	def _PyramidRoI_Feat(self, feat_maps, rois, im_info):
		''' roi pool on pyramid feature maps'''
		# do roi pooling based on predicted rois
		img_area = im_info[0][0] * im_info[0][1]
		h = rois.data[:, 4] - rois.data[:, 2] + 1
		w = rois.data[:, 3] - rois.data[:, 1] + 1

		### 不同尺度的ROI使用不同特征层作为ROI pooling层的输入
		# roi_level = torch.log(torch.sqrt(h * w) / 224.0)
		# roi_level = torch.round(roi_level + 4)
		# roi_level[roi_level < 2] = 2
		# roi_level[roi_level > 5] = 5

		# ###  直接选择P2特征层   2019.4.24.20:17
		roi_level = torch.log(torch.sqrt(h * w) / 100.0)
		roi_level = torch.round(roi_level + 4)
		roi_level[roi_level < 2] = 2
		roi_level[roi_level > 5] = 5

		# roi_level.fill_(5)
		if cfg.POOLING_MODE == 'crop':
			# pdb.set_trace()
			# pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
			# NOTE: need to add pyrmaid
			grid_xy = _affine_grid_gen(rois, feat_maps.size()[2:], self.grid_size)
			grid_yx = torch.stack([grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]], 3).contiguous()
			roi_pool_feat = self.RCNN_roi_crop(feat_maps, Variable(grid_yx).detach())
			if cfg.CROP_RESIZE_WITH_MAX_POOL:
				roi_pool_feat = F.max_pool2d(roi_pool_feat, 2, 2)

		elif cfg.POOLING_MODE == 'align':
			roi_pool_feats = []
			box_to_levels = []
			for i, l in enumerate(range(2, 6)):
				if (roi_level == l).sum() == 0:
					continue
				# idx_l = (roi_level == l).nonzero().squeeze()
				idx_l = (roi_level == l).nonzero()
				if idx_l.shape[0] > 1:
					idx_l = idx_l.squeeze()
				else:
					idx_l = idx_l.view(-1)
				box_to_levels.append(idx_l)
				scale = feat_maps[i].size(2) / im_info[0][0]
				feat = self.RCNN_roi_align(feat_maps[i], rois[idx_l], scale)
				roi_pool_feats.append(feat)
			roi_pool_feat = torch.cat(roi_pool_feats, 0)
			box_to_level = torch.cat(box_to_levels, 0)
			idx_sorted, order = torch.sort(box_to_level)
			roi_pool_feat = roi_pool_feat[order]

		elif cfg.POOLING_MODE == 'pool':
			roi_pool_feats = []
			box_to_levels = []
			for i, l in enumerate(range(2, 6)):
				if (roi_level == l).sum() == 0:
					continue
				idx_l = (roi_level == l).nonzero().squeeze()
				box_to_levels.append(idx_l)
				scale = feat_maps[i].size(2) / im_info[0][0]
				feat = self.RCNN_roi_pool(feat_maps[i], rois[idx_l], scale)
				roi_pool_feats.append(feat)
			roi_pool_feat = torch.cat(roi_pool_feats, 0)
			box_to_level = torch.cat(box_to_levels, 0)
			idx_sorted, order = torch.sort(box_to_level)
			roi_pool_feat = roi_pool_feat[order]

		return roi_pool_feat

	def forward(self, im_data, im_info, gt_boxes, num_boxes):
		batch_size = im_data.size(0)

		im_info = im_info.data
		gt_boxes = gt_boxes.data
		num_boxes = num_boxes.data

		# feed image data to base model to obtain base feature map
		# Bottom-up
		c1 = self.RCNN_layer0(im_data)
		c2 = self.RCNN_layer1(c1)
		c3 = self.RCNN_layer2(c2)
		c4 = self.RCNN_layer3(c3)
		c5 = self.RCNN_layer4(c4)
		# Top-down
		p5 = self.RCNN_toplayer(c5)
		p4 = self._upsample_add(p5, self.RCNN_latlayer1(c4))
		p4 = self.RCNN_smooth1(p4)
		p3 = self._upsample_add(p4, self.RCNN_latlayer2(c3))
		p3 = self.RCNN_smooth2(p3)
		p2 = self._upsample_add(p3, self.RCNN_latlayer3(c2))
		p2 = self.RCNN_smooth3(p2)

		p6 = self.maxpool2d(p5)

		rpn_feature_maps = [p2, p3, p4, p5, p6]
		mrcnn_feature_maps = [p2, p3, p4, p5]

		rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(rpn_feature_maps, im_info, gt_boxes, num_boxes)

		# feed image data to base model to obtain base feature map
		# Bottom-up
		c1_det = self.RCNN_layer0_det(im_data)
		c2_det = self.RCNN_layer1_det(c1)
		c3_det = self.RCNN_layer2_det(c2)
		c4_det = self.RCNN_layer3_det(c3)
		c5_det = self.RCNN_layer4_det(c4)
		c6_det = self.RCNN_layer5_det(c5)

		# Top-down
		p6_det = self.RCNN_toplayer_det(c6_det)
		p5_det = self.RCNN_latlayer1_det(c5_det) + p6_det
		p4_det = self.RCNN_latlayer2_det(c4_det) + p5
		p3_det = self._upsample_add(p4_det, self.RCNN_latlayer3(c3_det))
		p3_det = self.RCNN_smooth1_det(p3_det)
		p2_det = self._upsample_add(p3_det, self.RCNN_latlayer4(c2_det))
		p2_det = self.RCNN_smooth2_det(p2_det)

		rpn_feature_maps_det = [p2_det, p3_det, p4_det, p5_det, p6_det]
		mrcnn_feature_maps_det = [p2_det, p3_det, p4_det, p5_det]

		rois_det, rpn_loss_cls_det, rpn_loss_bbox_det = self.RCNN_rpn(rpn_feature_maps_det, im_info, gt_boxes, num_boxes)

		rpn_loss_cls = rpn_loss_cls + rpn_loss_cls_det
		rpn_loss_bbox = rpn_loss_bbox_det + rpn_loss_bbox


		Use_emsemble = False
		if Use_emsemble:
			model_ft = Cnn()
			# pretrained_model = '/home/lab30202/lq/ai_future/low_exporsure_time/low_exporsure_v6/data/pretrained_model/galxay_star_classification_vgg.pth'
			# state_dict = torch.load(pretrained_model)
			# model_ft.load_state_dict({k:v for k,v in state_dict.items() if k in model_ft.state_dict()})
			model_ft = model_ft.cuda()
			feature_map = model_ft.convnet(im_data)

		# if it is training phrase, then use ground trubut bboxes for refining
		if self.training:
			roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
			rois, rois_label, gt_assign, rois_target, rois_inside_ws, rois_outside_ws = roi_data

			roi_data_det = self.RCNN_proposal_target(rois_det, gt_boxes, num_boxes)
			rois_det, rois_label_det, gt_assign_det, rois_target_det, rois_inside_ws_det, rois_outside_ws_det = roi_data_det

			## NOTE: additionally, normalize proposals to range [0, 1],
			#        this is necessary so that the following roi pooling
			#        is correct on different feature maps
			# rois[:, :, 1::2] /= im_info[0][1]
			# rois[:, :, 2::2] /= im_info[0][0]

			rois = rois.view(-1, 5)
			rois_label = rois_label.view(-1).long()
			gt_assign = gt_assign.view(-1).long()
			pos_id = rois_label.nonzero().squeeze()
			gt_assign_pos = gt_assign[pos_id]
			rois_label_pos = rois_label[pos_id]
			rois_label_pos_ids = pos_id

			rois_pos = Variable(rois[pos_id])
			rois = Variable(rois)
			rois_label = Variable(rois_label)

			rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
			rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
			rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))

			rois_det = rois_det.view(-1, 5)
			rois_label_det = rois_label.view(-1).long()
			gt_assign_det = gt_assign_det.view(-1).long()
			pos_id_det = rois_label_det.nonzero().squeeze()
			gt_assign_pos_det = gt_assign_det[pos_id_det]
			rois_label_pos_det = rois_label_det[pos_id_det]
			rois_label_pos_ids_det = pos_id_det

			rois_pos_det = Variable(rois[pos_id_det])
			rois_det = Variable(rois_det)
			rois_label_det = Variable(rois_label_det)

			rois_target_det = Variable(rois_target_det.view(-1, rois_target_det.size(2)))
			rois_inside_ws_det = Variable(rois_inside_ws_det.view(-1, rois_inside_ws_det.size(2)))
			rois_outside_ws_det = Variable(rois_outside_ws_det.view(-1, rois_outside_ws_det.size(2)))
		else:
			## NOTE: additionally, normalize proposals to range [0, 1],
			#        this is necessary so that the following roi pooling
			#        is correct on different feature maps
			# rois[:, :, 1::2] /= im_info[0][1]
			# rois[:, :, 2::2] /= im_info[0][0]

			rois_label = None
			gt_assign = None
			rois_target = None
			rois_inside_ws = None
			rois_outside_ws = None
			rpn_loss_cls = 0
			rpn_loss_bbox = 0
			rois = rois.view(-1, 5)
			pos_id = torch.arange(0, rois.size(0)).long().type_as(rois).long()
			rois_label_pos_ids = pos_id
			rois_pos = Variable(rois[pos_id])
			rois = Variable(rois)

			rois_label_det = None
			gt_assign_det = None
			rois_target_det = None
			rois_inside_ws_det = None
			rois_outside_ws_det = None
			rpn_loss_cls_det = 0
			rpn_loss_bbox_det = 0
			rois_det = rois_det.view(-1, 5)
			pos_id_det = torch.arange(0, rois_det.size(0)).long().type_as(rois).long()
			rois_label_pos_ids_det = pos_id_det
			rois_pos_det = Variable(rois_det[pos_id_det])
			rois_det = Variable(rois_det)

		# pooling features based on rois, output 14x14 map   (128,64,7,7)
		roi_pool_feat = self._PyramidRoI_Feat(mrcnn_feature_maps, rois, im_info)
		roi_pool_feat_det = self._PyramidRoI_Feat(mrcnn_feature_maps_det, rois_det, im_info)

		###  ensemable learning add by lq : 2019.6.24 , use mrcnn_feature_maps
		if Use_emsemble:
			if self.training:
				idx_l = [x for x in range(0, 128, 1)]
			else:
				idx_l = [x for x in range(0, 300, 1)]
			idx_l = torch.LongTensor(idx_l)
			scale = 0.5
			feat = self.RCNN_roi_align(feature_map, rois[idx_l], scale)
			out_roi_pool = feat.view(feat.shape[0], -1)
			cls_score_vgg = model_ft.fc(out_roi_pool)
		# cls_prob_vgg = F.softmax(cls_score_vgg,dim=1)

		# feed pooled features to top model (128,256)输入到分类全连接
		pooled_feat = self._head_to_tail(roi_pool_feat)
		pooled_feat_det = self._head_to_tail(roi_pool_feat_det)

		# compute bbox offset
		bbox_pred = self.RCNN_bbox_pred(pooled_feat)
		bbox_pred_det = self.RCNN_bbox_pred(pooled_feat_det)
		if self.training and not self.class_agnostic:
			# select the corresponding columns according to roi labels
			bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
			bbox_pred_select = torch.gather(bbox_pred_view, 1,
			                                rois_label.long().view(rois_label.size(0), 1, 1).expand(rois_label.size(0),
			                                                                                        1, 4))
			bbox_pred = bbox_pred_select.squeeze(1)

			bbox_pred_view_det = bbox_pred_det.view(bbox_pred_det.size(0), int(bbox_pred_det.size(1) / 4), 4)
			bbox_pred_select_det = torch.gather(bbox_pred_view_det, 1,
			                                rois_label_det.long().view(rois_label_det.size(0), 1, 1).expand(rois_label_det.size(0),
			                                                                                        1, 4))
			bbox_pred_det = bbox_pred_select_det.squeeze(1)


		# compute object classification probability
		cls_score = self.RCNN_cls_score(pooled_feat)
		cls_score_det = self.RCNN_cls_score(pooled_feat_det)
		# cls_prob = F.softmax(cls_score,dim=1)

		if Use_emsemble:
			cls_score_liner = cls_score + cls_score_vgg
			cls_score = model_ft.fc_new(cls_score_liner)
			cls_prob = F.softmax(cls_score, dim=1)
		# cls_score_all = cls_prob + cls_prob_vgg
		# cls_prob = F.softmax(cls_score_all,dim=1)
		else:
			cls_score_11 = cls_score + cls_score_det
			cls_score = model_ft.fc_new(cls_score_11)
			cls_prob = F.softmax(cls_score, dim=1)

		RCNN_loss_cls = 0
		RCNN_loss_bbox = 0

		if self.training:
			# loss (cross entropy) for object classification
			Use_focal_loss = True
			if not Use_focal_loss:
				RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
			else:
				FL = FocalLoss(class_num=self.n_classes, alpha=0.25, gamma=2)
				RCNN_loss_cls = FL(cls_score, rois_label)
				RCNN_loss_cls = RCNN_loss_cls.type(torch.FloatTensor).cuda()

			# loss (l1-norm) for bounding box regression
			RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)
			RCNN_loss_bbox_det = _smooth_l1_loss(bbox_pred_det, rois_target_det, rois_inside_ws_det, rois_outside_ws_det)


		rois = rois.view(batch_size, -1, rois.size(1))
		cls_prob = cls_prob.view(batch_size, -1, cls_prob.size(1))
		bbox_pred = bbox_pred.view(batch_size, -1, bbox_pred.size(1))

		if self.training:
			rois_label = rois_label.view(batch_size, -1)
			rpn_loss_cls = torch.unsqueeze(rpn_loss_cls, 0)
			rpn_loss_bbox = torch.unsqueeze(rpn_loss_bbox, 0)
			RCNN_loss_cls = torch.unsqueeze(RCNN_loss_cls, 0)
			RCNN_loss_bbox = torch.unsqueeze(RCNN_loss_bbox, 0)

		return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label



