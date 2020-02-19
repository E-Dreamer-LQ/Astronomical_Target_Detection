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
from model.AlignPool.roi_util.roi_align import ROIAlign as RoIAlignAvg
from model.AlignPool.roi_util.roi_pool import ROIPool as _RoIPooling
from model.rpn.proposal_target_layer import _ProposalTargetLayer
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta
from model.utils.net_utils import FocalLoss
from model.fpn.Vgg_mine import Cnn
from model.fpn.detnet_backbone import Detnet
from model.utils.label_smooth import LabelSmoothSoftmaxCE
from model.utils.generalized_iou_loss import generalized_iou_loss
from model.utils.Giou_loss import Giou_np
from model.rpn.bbox_transform import bbox_transform_inv
from model.rpn.bbox_transform import clip_boxes
import time
import pdb


class _FPN_res(nn.Module):
    """ FPN """

    def __init__(self, classes, class_agnostic):
        super(_FPN_res, self).__init__()
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
        self.RCNN_roi_pool = _RoIPooling((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)   
        self.RCNN_roi_align = RoIAlignAvg((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0 , 2)  
        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE

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

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)
        weights_init(self.RCNN_top, 0, 0.01, cfg.TRAIN.TRUNCATED)

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

        roi_level = torch.log(torch.sqrt(h * w) / 50.0)
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

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, gt_assign, rois_target, rois_inside_ws, rois_outside_ws = roi_data

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

        # pooling features based on rois, output 14x14 map   (128,64,7,7)
        roi_pool_feat = self._PyramidRoI_Feat(mrcnn_feature_maps, rois, im_info)

        Use_emsemble = False
        emsemble_vgg, emsemble_detnet = [False, True]
        if Use_emsemble:
            if emsemble_vgg:
                model_vgg = Cnn()
                model_vgg = model_vgg.cuda()
                ## vgg net
                pretrained_model_vgg = '/home/lab30202/lq/ai_future/single_classsification_vgg/model_save/galxay_star_classification_vgg.pth'  # 预训练模型参数保存地址
                pretrained_dict = torch.load(pretrained_model_vgg)
                model_dict = model_vgg.state_dict()  
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}  
                model_dict.update(pretrained_dict)  
                model_vgg.load_state_dict(model_dict)  
                feature_map_vgg = model_vgg.convnet(im_data)
                if self.training:
                    idx_l = [x for x in range(0, 128, 1)]
                else:
                    idx_l = [x for x in range(0, 300, 1)]
                idx_l = torch.LongTensor(idx_l)
                feat = self.RCNN_roi_align(feature_map_vgg, rois[idx_l], 0.5)
                roi_pool_vgg = feat.view(feat.shape[0], -1)
                cls_score_vgg = model_vgg.fc(roi_pool_vgg)
                # cls_prob_vgg = F.softmax(cls_score_vgg,dim=1)
            if emsemble_detnet:
                ## detnet
                detnet = Detnet()
                detnet = detnet.cuda()
                # Bottom-up
                c1_det = detnet.RCNN_layer0_det(im_data)
                c2_det = detnet.RCNN_layer1_det(c1_det)
                c3_det = detnet.RCNN_layer2_det(c2_det)
                c4_det = detnet.RCNN_layer3_det(c3_det)
                c5_det = detnet.RCNN_layer4_det(c4_det)
                c6_det = detnet.RCNN_layer5_det(c5_det)

                # Top-down
                p6_det = detnet.RCNN_toplayer_det(c6_det)
                p5_det = detnet.RCNN_latlayer1_det(c5_det) + p6_det
                p4_det = detnet.RCNN_latlayer2_det(c4_det) + p5_det
                p3_det = detnet._upsample_add(p4_det, detnet.RCNN_latlayer3_det(c3_det))
                p3_det = detnet.RCNN_smooth1_det(p3_det)
                p2_det = detnet._upsample_add(p3_det, detnet.RCNN_latlayer4_det(c2_det))
                p2_det = detnet.RCNN_smooth2_det(p2_det)

                rpn_feature_maps_det = [p2_det, p3_det, p4_det, p5_det, p6_det]
                mrcnn_feature_maps_det = [p2_det, p3_det, p4_det, p5_det]
                rois_det, rpn_loss_cls_det, rpn_loss_bbox_det = self.RCNN_rpn(rpn_feature_maps_det, im_info, gt_boxes,
                                                                              num_boxes)
                if self.training:
                    roi_data_det = self.RCNN_proposal_target(rois_det, gt_boxes, num_boxes)
                    rois_det, rois_label_det, gt_assign_det, rois_target_det, rois_inside_ws_det, rois_outside_ws_det = roi_data_det
                    rois_det = rois_det.view(-1, 5)
                    rois_label_det = rois_label_det.view(-1).long()
                    gt_assign_det = gt_assign_det.view(-1).long()
                    pos_id_det = rois_label_det.nonzero().squeeze()
                    gt_assign_pos_det = gt_assign_det[pos_id_det]
                    rois_label_pos_det = rois_label_det[pos_id_det]
                    rois_label_pos_ids_det = pos_id_det

                    rois_pos_det = Variable(rois_det[pos_id_det])
                    rois_det = Variable(rois_det)
                    rois_label_det = Variable(rois_label_det)

                    rois_target_det = Variable(rois_target_det.view(-1, rois_target_det.size(2)))
                    rois_inside_ws_det = Variable(rois_inside_ws_det.view(-1, rois_inside_ws_det.size(2)))
                    rois_outside_ws_det = Variable(rois_outside_ws_det.view(-1, rois_outside_ws_det.size(2)))
                else:
                    rois_label_det = None
                    gt_assign_det = None
                    rois_target_det = None
                    rois_inside_ws_det = None
                    rois_outside_ws_det = None
                    rpn_loss_cls_det = 0
                    rpn_loss_bbox_det = 0
                    rois_det = rois_det.view(-1, 5)
                    pos_id_det = torch.arange(0, rois_det.size(0)).long().type_as(rois_det).long()
                    rois_label_pos_ids_det = pos_id_det
                    rois_pos_det = Variable(rois_det[pos_id_det])
                    rois_det = Variable(rois_det)

                feat_det = self._PyramidRoI_Feat(mrcnn_feature_maps_det, rois, im_info)
                if emsemble_detnet:
                    pooled_feat_det = detnet._head_to_tail(feat_det)
                    cls_score_det = self.RCNN_cls_score(pooled_feat_det)
                else:
                    roi_pool_det = feat_det.view(feat_det.shape[0], -1)
                    cls_score_det = model_vgg.fc(roi_pool_det)

        pooled_feat = self._head_to_tail(roi_pool_feat)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1,
                                            rois_label.long().view(rois_label.size(0), 1, 1).expand(rois_label.size(0),
                                                                                                    1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        # cls_prob = F.softmax(cls_score,dim=1)

        if Use_emsemble:
            if emsemble_detnet and emsemble_vgg:
                cls_score_liner = 0.5 * cls_score + 0.3 * cls_score_vgg + 0.2 * cls_score_det
                cls_score = model_vgg.fc_new(cls_score_liner)
                cls_prob = F.softmax(cls_score, dim=1)
            elif emsemble_vgg and not emsemble_detnet:
                cls_score_liner = cls_score + cls_score_vgg
                cls_score = model_vgg.fc_new(cls_score_liner)
                cls_prob = F.softmax(cls_score, dim=1)
            elif emsemble_detnet and not emsemble_vgg:
                cls_score_liner = cls_score + cls_score_det
                cls_score = detnet.fc_add(cls_score_liner)
                cls_prob = F.softmax(cls_score, dim=1)
        else:
            cls_score = self.RCNN_cls_score(pooled_feat)
            cls_prob = F.softmax(cls_score, dim=1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # loss (cross entropy) for object classification
            Use_focal_loss = True
            Use_label_smoothing = False
            Use_Giou_loss = False
            if not Use_focal_loss:
                if Use_label_smoothing:
                    # criteria = LabelSmoothSoftmaxCE(label_smoothing=0.1)
                    criteria = LabelSmoothSoftmaxCE(lb_pos=0.9, lb_neg=5e-3)
                    RCNN_loss_cls = criteria(cls_score, rois_label)
                else:
                    RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
            else:
                FL = FocalLoss(class_num=self.n_classes, alpha=1, gamma=2)
                RCNN_loss_cls = FL(cls_score, rois_label)
                RCNN_loss_cls = RCNN_loss_cls.type(torch.FloatTensor).cuda()

            # loss (l1-norm) for bounding box regression
            if Use_Giou_loss:
                rois1 = rois.view(batch_size, -1, rois.size(1))
                boxes = rois1.data[:, :, 1:5]
                bbox_pred1 = bbox_pred.view(batch_size, -1, bbox_pred.size(1))
                box_deltas = bbox_pred1.data
                # if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                #     # Optionally normalize targets by a precomputed mean and stdev
                #     box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                #                  + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                #     box_deltas = box_deltas.view(1, -1, 4 * len(self.classes))

                pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
                pred_boxes /= im_info[0][2].cuda()
                # RCNN_loss_bbox = generalized_iou_loss(rois_target,bbox_pred)
                _, _, RCNN_loss_bbox = Giou_np(pred_boxes, boxes)
            else:
                RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

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

