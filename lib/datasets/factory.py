# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append('..')

__sets = {}
from datasets.star import star
import numpy as np


# set up image net.
for split in ['train','test']:
    name = '{}data300-300'.format(split)
    data_path = '/home/lab30202/sdb/liuqiang/2020-2-11-star_detection_release'
    __sets[name] = (lambda split=split, data_path=data_path: star(split,data_path))

def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())


if  __name__ == '__main__':
    for split in ['train','test']:
        name = '{}data300-300'.format(split)
        print("name:",name)
        imdb = get_imdb(name)
        print("imdb:",imdb)
    print(list_imdbs())


