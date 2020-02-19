import cv2
import numpy as np
import os
import glob


def filt(list_img):
    number_detect = list_img.shape[0]
    for i in range(number_detect):
        x_axis = list_img[i][1]
        y_axis = list_img[i][2]





save_path = "/home/lab30202/lq/ai_future/low_exporsure_emsemble/fpn_resnet_detnet_v2/validation"

list_list = glob.glob(os.path.join(save_path,"*.fits.txt"))
for single_list in list_list:
    list_img = np.loadtxt(single_list)
    image_index = single_list.split("/")[-1].split(".fits.txt")[0]
    filt_img = filt(list_img)




