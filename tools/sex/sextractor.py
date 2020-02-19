#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:55:12 2019

@author: liuqiang
"""

import os
import glob 
import csv 
import numpy as np
import subprocess

save_as_fits_path = '/home/lab30202/sdb/liuqiang/2019-11-10-Real_data/fpn_v1/validation/sex'
sexfile_name = '/home/lab30202/sdb/liuqiang/2019-11-10-Real_data/fpn_v1/validation/sex/default.sex'

def gain_fits_data(path):
    fitfilepath = path+'/*.fit*'
    fitfile_name = glob.glob(fitfilepath)
    return fitfile_name 

def sextract(fits_name,default_sex):
    process = subprocess.Popen('sex %s -c %s' % (fits_name, default_sex), shell=True)
    process.wait()
    
def pos_file_alter(sexfile_name,new_name):
    ## at first: read the txt file
    lines=[]
    with open(sexfile_name,'r') as f:
        lines=f.readlines()
    lines[6] = 'CATALOG_NAME     %s.txt     # name of the output catalog'%new_name + '\n'
    
    with open(sexfile_name,'w') as f:
        for data in lines:
            f.write(data)
        f.flush()
        
fitfile_name_all = gain_fits_data(save_as_fits_path)
sexfile_name = sexfile_name
for single_fit in fitfile_name_all:
    alter_name = single_fit.split("/")[-1]
#    .split(".fits")[0]
    pos_file_alter(sexfile_name,alter_name)
    sextract(alter_name,sexfile_name)
print("completed!!!")
    





