# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
import numpy as np
import scipy.sparse
import pickle as cPickle
import math
import glob
import scipy.io as sio
from astropy.io import fits
import sys
sys.path.append('..')
from datasets.imdb_v1 import imdb
from model.utils.config import cfg
from datasets.imdb_v1 import ROOT_DIR
from datasets import  ds_utils
from datasets.voc_eval import voc_eval
# TODO: make fast_rcnn irrelevant
# >>>> obsolete, because it depends on sth outside of this project


class star(imdb):
    def __init__(self, image_set, data_path):
        imdb.__init__(self,image_set +'data300-300')
        self._image_set = image_set
        self._data_path = data_path
        self._devkit_path = '/home/lab30202/sdb/liuqiang/2020-2-11-star_detection_release/detection_1.0_v2/fpn_v1/data'
        self._classes = ('__background__',  # always index 0
                        'star','galaxy')
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._image_ext = '.fit*'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        # self._roidb_handler = self.selective_search_roidb
        self._roidb_handler = self.gt_roidb
        self.fitsfilelist=glob.glob(self._data_path+'/'+self._image_set +'data300-300/'+'*.fit*')
        self.positionfilelist=glob.glob(self._data_path+'/'+self._image_set +'data300-300/'+'*.list')

        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return i

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path,index)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        image_set_path = self._data_path
        assert os.path.exists(image_set_path), \
            'Path does not exist: {}'.format(image_set_path)
        image_index = glob.glob(image_set_path+'/'+self._image_set +'data300-300/'+'*.fit*')
        image_index.sort()

        return image_index


    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print ('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_star_txt(index)
                    for index in self._image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print ('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def rpn_roidb(self):
        if self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print ('loading {}'.format(filename))
        assert os.path.exists(filename), \
            'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
            'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in range(raw_data.shape[0]):
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
            keep = ds_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_star_txt(self,indx):         ##  change from _load_pascal_annotation(self,indx)
        name_all = []
        name_all.append(indx)

        base_name = os.path.basename(indx).split('.')[0]
        starposfilenametmp = self._data_path+'/'+self._image_set +'data300-300/'+ base_name+ '.list'
        array_from_txt  = np.loadtxt(starposfilenametmp)
        objs_all_list = []
        max_value = 495
        for step, value_single_row in enumerate(array_from_txt):
            if 5 < array_from_txt[step][1] < max_value and 5 < array_from_txt[step][2] < max_value:  ##  remove nagative coordinate
                objs_all_list.append(value_single_row)
        objs_all = np.array(objs_all_list)
        num_objs = objs_all.shape[0]
        boxes = np.zeros((num_objs, 4), dtype=np.int32)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)    ##self.num_classes = 2
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        ishards = np.zeros((num_objs), dtype=np.int32)
        for ix in range(num_objs):
            if objs_all[ix][0] == 100:
                padding = 5
                x1 = objs_all[ix][1] - padding
                x2 = objs_all[ix][1] + padding
                y1 = objs_all[ix][2] - padding
                y2 = objs_all[ix][2] + padding
            else:
                padding = 6
                x1 = objs_all[ix][1] - padding
                x2 = objs_all[ix][1] + padding
                y1 = objs_all[ix][2] - padding
                y2 = objs_all[ix][2] + padding

            x1 = max(float(x1), 0)
            x2 = max(float(x2), 0)
            y1 = max(float(y1), 0)
            y2 = max(float(y2), 0)
            ishards[ix] = 0
            if objs_all[ix][0] == 100:
                cls = self._class_to_ind['star']
            else:
                cls = self._class_to_ind['galaxy']
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1) * (y2 - y1)   ## default = 10*10
        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_ishard': ishards,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas,
                }

    def _get_voc_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        # filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'

        filename = 'det_' + self._image_set + '_{:s}.txt'

        filedir = os.path.join(self._devkit_path, 'results', 'star_dectect', 'Main')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path


    def _write_voc_results_file(self, all_boxes):
        '''
        根据不同类别写result file，文件中遍历不同图像，所有的boxes，
        一次写入一个框的img_id score x0+1 y0+1 x1+1 y1+1 六个数
        '''
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print ('Writing {} VOC results file'.format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir='output'):
        '''
        根据上面的result file，对不同类别的结果进行评估，写入output文件
        filename表示_write_voc_results_file得到的不同类别对应的文件
        annopath表示ground-truth的xml文件
        imagesetfile表示测试的图像列表
        cls是类别
        cachedir根据imagesetfile图片名解析annopath－xml文件得到的文件，文件格式是每个图像名对应图像中的boxes和类别等，如不存在会在voc_eval函数生成。
        ovthresh overlap的阈值
        '''
        imagesetfile_path =  self._data_path+'/'+self._image_set +'data300-300/'+'*.fit*'
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        recs = {}
        precs = {}
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)

            ##  cls is the class name we predict
            rec, prec, ap = voc_eval(
                filename,imagesetfile_path, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric)
            recs[cls] = rec
            precs[cls] = prec
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            print("rec:",rec)
            print("prec:",prec)
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')

        return recs,precs,np.mean(aps)


    def evaluate_detections(self, all_boxes, output_dir):
        self._write_voc_results_file(all_boxes)
        recs,precs,mean_ap = self._do_python_eval(output_dir)
        for cls in self._classes:
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            os.remove(filename)

        return recs,precs,mean_ap


if __name__ == '__main__':
    d = star('train', '/home/lab30202/sdb/liuqiang/2019-11-10-Real_data')
    res = d.roidb
    from IPython import embed;
    embed()


