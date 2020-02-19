import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
plt.rcParams['figure.dpi'] = 300 #分辨率
plt.rc('font',family='Times New Roman')   ###  设置新字体
file_path = '/home/lab30202/lq/ai_future/low_exporsure_emsemble/fpn_resnet_detnet_v2/validation'
thresholds = [0.4,0.5,0.6]

def sex_list_match(sex_img,list_img,section_target_num):
    ### matching
    match_num = 0
    match = 0
    det_target = sex_img.shape[0]
    number_target = section_target_num
    match_value = 1.5
    match_list = []
    for i in range(det_target):
        match_single = 0
        x_axis = sex_img[i][1]
        y_axis = sex_img[i][2]
        for j in range(number_target):
            if abs(x_axis - list_img[j][1]) <= match_value and abs(y_axis - list_img[j][2]) <= match_value:
                match_num += 1
                match += 1
                match_single += 1
        match_list.append(match_single)
        if match_num > number_target:
            match_num =  number_target
    precision = match_num / det_target
    if number_target != 0:
        recall = match/number_target
    else:
        recall = 1
        precision = 1
    return precision,recall


def det_list_match(det_img,list_img,section_target_num):
    ### matching
    match_num = 0
    match = 0
    det_target = det_img.shape[0]
    number_target = section_target_num
    match_value = 1.5
    for i in range(det_target):
        x_axis = (det_img[i][0]+det_img[i][2])/2
        y_axis = (det_img[i][1]+det_img[i][3])/2
        for j in range(number_target):
            if abs(x_axis - list_img[j][1]) <= match_value and abs(y_axis - list_img[j][2]) <= match_value:
                match_num += 1
                match += 1
        if match_num > number_target:
            match_num = number_target
    precision = match_num/det_target
    if number_target != 0:
        recall = match/number_target
    else:
        recall = 1
        precision = 1

    return precision,recall
def compute_section_target(list_list_img):
    mag9_10 = 0
    mag10_11 = 0
    mag11_12 = 0
    mag12_13 = 0
    mag13_14 = 0
    mag14_15 = 0
    mag15_16 = 0
    mag16_17 = 0
    mag17_18 = 0
    mag18_19 = 0
    mag19_20 = 0
    mag20_21 = 0
    mag21_22 = 0
    mag22_23 = 0
    mag23_24 = 0
    mag24_25 = 0
    mag9_10_list = []
    mag10_11_list = []
    mag11_12_list = []
    mag12_13_list = []
    mag13_14_list = []
    mag14_15_list = []
    mag15_16_list = []
    mag16_17_list = []
    mag17_18_list = []
    mag18_19_list = []
    mag19_20_list = []
    mag20_21_list = []
    mag21_22_list = []
    mag22_23_list = []
    mag23_24_list = []
    mag24_25_list = []

    for step,value in enumerate(list_list_img):
        if 9 <=  value[3] < 10:
            mag9_10 += 1
            mag9_10_list.append(value)
        if 10 <=  value[3] < 11:
            mag10_11 += 1
            mag10_11_list.append(value)
        if 11 <=  value[3] < 12:
            mag11_12 += 1
            mag11_12_list.append(value)
        if 12 <=  value[3] < 13:
            mag12_13 += 1
            mag12_13_list.append(value)
        if 13 <=  value[3] < 14:
            mag13_14 += 1
            mag13_14_list.append(value)
        if 14 <=  value[3] < 15:
            mag14_15 += 1
            mag14_15_list.append(value)
        elif 15 <=  value[3] < 16:
            mag15_16 += 1
            mag15_16_list.append(value)
        elif 16 <=  value[3] < 17:
            mag16_17 += 1
            mag16_17_list.append(value)
        elif 17 <=  value[3] < 18:
            mag17_18 += 1
            mag17_18_list.append(value)
        elif 18 <=  value[3] < 19:
            mag18_19 += 1
            mag18_19_list.append(value)
        elif 19 <=  value[3] < 20:
            mag19_20 += 1
            mag19_20_list.append(value)
        elif 20 <=  value[3] < 21:
            mag20_21 += 1
            mag20_21_list.append(value)
        elif 21 <=  value[3] < 22:
            mag21_22 += 1
            mag21_22_list.append(value)
        elif 22 <=  value[3] < 23:
            mag22_23 += 1
            mag22_23_list.append(value)
        elif 23 <=  value[3] < 24:
            mag23_24 += 1
            mag23_24_list.append(value)
        elif 24 <=  value[3] < 25:
            mag24_25 += 1
            mag24_25_list.append(value)
    mag9_10_list = np.array(mag9_10_list)
    mag10_11_list = np.array(mag10_11_list)
    mag11_12_list = np.array(mag11_12_list)
    mag12_13_list = np.array(mag12_13_list)
    mag13_14_list = np.array(mag13_14_list)

    mag14_15_list = np.array(mag14_15_list)
    mag15_16_list = np.array(mag15_16_list)
    mag16_17_list = np.array(mag16_17_list)
    mag17_18_list = np.array(mag17_18_list)
    mag18_19_list = np.array(mag18_19_list)
    mag19_20_list = np.array(mag19_20_list)
    mag20_21_list = np.array(mag20_21_list)
    mag21_22_list = np.array(mag21_22_list)
    mag22_23_list = np.array(mag22_23_list)
    mag23_24_list = np.array(mag23_24_list)
    mag24_25_list = np.array(mag24_25_list)
    return 	mag9_10,mag10_11,mag11_12,mag12_13,mag13_14,mag14_15 ,mag15_16 ,mag16_17,mag17_18,mag18_19,\
            mag19_20,mag20_21,mag21_22,mag22_23,mag23_24,mag24_25,\
            mag9_10_list,mag10_11_list,mag11_12_list,mag12_13_list,mag13_14_list,mag14_15_list,mag15_16_list,mag16_17_list,mag17_18_list,mag18_19_list,\
            mag19_20_list,mag20_21_list,mag21_22_list,\
            mag22_23_list,mag23_24_list,mag24_25_list


precision_all_sex = np.zeros((16))
recall_all_sex = np.zeros((16))
precision_all_det = np.zeros((16))
recall_all_det = np.zeros((16))
f1_score_all_sex = np.zeros(16)
f1_score_all_det = np.zeros(16)
f2_score_all_sex = np.zeros(16)
f2_score_all_det = np.zeros(16)




for threhold in thresholds:
    det_list = glob.glob(os.path.join(file_path, "*det_" + str(threhold) + ".txt"))
    list_list = glob.glob(os.path.join(file_path, '*.list'))
    fits_list = glob.glob(os.path.join(file_path, "*fits.txt"))
    det_file_num = len(det_list)
    list_file_num = len(list_list)
    fits_file_num = len(fits_list)
    if (det_file_num == list_file_num) and  (fits_file_num == list_file_num):
        for count,single in enumerate(det_list):
            name_index = single.split("/")[-1].split("_det_"+str(threhold))[0]
            fits_txt_name = os.path.join(file_path,name_index+'.fits.txt')
            list_name = os.path.join(file_path,name_index+'.list')
            det_img = np.loadtxt(single)
            sex_img = np.loadtxt(fits_txt_name)
            list_img = np.loadtxt(list_name)

            mag9_10, mag10_11, mag11_12, mag12_13, mag13_14, mag14_15, mag15_16, mag16_17, mag17_18, mag18_19, \
            mag19_20, mag20_21, mag21_22, mag22_23, mag23_24, mag24_25, \
            mag9_10_list, mag10_11_list, mag11_12_list, mag12_13_list, mag13_14_list, mag14_15_list, mag15_16_list, mag16_17_list, mag17_18_list, mag18_19_list, \
            mag19_20_list, mag20_21_list, mag21_22_list, \
            mag22_23_list, mag23_24_list, mag24_25_list = compute_section_target(list_img)


            mag_all = [mag9_10, mag10_11, mag11_12, mag12_13, mag13_14, mag14_15, mag15_16, mag16_17, mag17_18, mag18_19, \
            mag19_20, mag20_21, mag21_22, mag22_23, mag23_24, mag24_25]
            mag_all_list = [mag9_10_list, mag10_11_list, mag11_12_list, mag12_13_list, mag13_14_list, mag14_15_list, mag15_16_list, mag16_17_list, mag17_18_list, mag18_19_list, \
            mag19_20_list, mag20_21_list, mag21_22_list, \
            mag22_23_list, mag23_24_list, mag24_25_list, mag24_25_list]


            precision_sex = []
            recall_sex = []
            precision_det = []
            recall_det = []
            f1_score_sex = []
            f1_score_det = []
            f2_score_sex = []
            f2_score_det = []

            mag_nums_all =  sum(mag_all)

            for step, value in enumerate(mag_all):
                list_list_img_index = mag_all_list[step]
                prec_sex, rec_sex = sex_list_match(sex_img, list_list_img_index, value)
                if value != 0:
                    prec_sex = prec_sex * (mag_nums_all/value)
                    if prec_sex > 1:
                        prec_sex =  1
                else:
                    prec_sex = 1
                ### recompute the precision
                if prec_sex != 0 and rec_sex != 0:
                    f1_score_s = 2*prec_sex*rec_sex/(prec_sex+rec_sex)
                    f2_score_s = (1+2*2) * prec_sex * rec_sex / (2*2*prec_sex + rec_sex)
                else:
                    f1_score_s = 0
                    f2_score_s = 0
                f1_score_sex.append(f1_score_s)
                f2_score_sex.append(f2_score_s)
                precision_sex.append(prec_sex)
                recall_sex.append(rec_sex)
                prec_det, rec_det = det_list_match(det_img, list_list_img_index, value)
                if value != 0:
                    prec_det = prec_det * (mag_nums_all/value)
                    if prec_det > 1:
                        prec_det =  1
                else:
                    prec_det = 1
                precision_det.append(prec_det)
                recall_det.append(rec_det)
                if prec_det != 0 and rec_det != 0:
                    f1_score_d = 2*prec_det*rec_det/(prec_det+rec_det)
                    f2_score_d = (1 + 2 * 2) * prec_det * rec_det / (2 * 2 * prec_det + rec_det)
                else:
                    f1_score_d = 0
                    f2_score_d = 0
                f1_score_det.append(f1_score_d)
                f2_score_det.append(f2_score_d)
            precision_sex = np.array(precision_sex)
            recall_sex = np.array(recall_sex)
            precision_det = np.array(precision_det)
            recall_det = np.array(recall_det)
            f1_score_sex = np.array(f1_score_sex)
            f1_score_det = np.array(f1_score_det)
            f2_score_sex = np.array(f2_score_sex)
            f2_score_det = np.array(f2_score_det)

            precision_all_sex += precision_sex
            recall_all_sex += recall_sex
            precision_all_det += precision_det
            recall_all_det += recall_det
            f1_score_all_sex += f1_score_sex
            f1_score_all_det += f1_score_det
            f2_score_all_sex += f2_score_sex
            f2_score_all_det += f2_score_det

        precision_all_sex = precision_all_sex/(count+1)
        recall_all_sex = recall_all_sex/(count+1)
        precision_all_det = precision_all_det/(count+1)
        recall_all_det = recall_all_det/(count+1)
        f1_score_all_sex = f1_score_all_sex/(count+1)
        f1_score_all_det = f1_score_all_det/(count+1)
        f2_score_all_sex = f2_score_all_sex/(count+1)
        f2_score_all_det = f2_score_all_det/(count+1)


        x = np.arange(10.,26.,1)
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(x, recall_all_sex,linewidth = '2',color='yellow',label="recall_sex")
        ax1.plot(x, recall_all_det,linewidth = '2', color='green', label="recall_det")
        a = ax1.legend(loc="upper right")
        x_major_locator=MultipleLocator(1)
        ax1.xaxis.set_major_locator(x_major_locator)
        Have_mag24_25 = False
        if Have_mag24_25:
            ax1.set_xlim([9.,25.])
        else:
            ax1.set_xlim([9., 24.])
        ax1.set_ylim([0.,1.1])
        ax1.set_xlabel('Magnitude', fontsize=16)
        ax1.set_ylabel('Recall',fontsize=16)
        ax1.set_title("Recall-mag-Precision",fontsize=16)
        ax2 = ax1.twinx()  # this is the important function
        ax2.plot(x, precision_all_sex, linewidth = '2',color='red',label="precision_sex")
        ax2.plot(x, precision_all_det, linewidth = '2',color='black', label="precision_det")
        b = ax2.legend(loc="lower left")
        ax2.set_ylim([0,1.1])
        ax2.set_ylabel('Precision',fontsize=16)
        ax2.set_xlabel('Same X for both Recall and Precision',fontsize=16)
        plt.savefig('/home/lab30202/lq/ai_future/low_exporsure_emsemble/fpn_resnet_detnet_v2/validation/match_visualization/sex_det_{}.jpg'.format(threhold))
        plt.close()

        # f1-score
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(x, f1_score_all_sex, linewidth = '2',color='yellow', label="f1_score_sex")
        ax1.plot(x, f1_score_all_det, linewidth = '2',color='green', label="f1_score_det")
        ax1.plot(x, f2_score_all_sex, linewidth = '2',color='red', label="f2_score_sex")
        ax1.plot(x, f2_score_all_det, linewidth = '2',color='black', label="f2_score_det")
        a = ax1.legend(loc="lower left")
        x_major_locator = MultipleLocator(1)
        ax1.xaxis.set_major_locator(x_major_locator)
        Have_mag24_25 = False
        if Have_mag24_25:
            ax1.set_xlim([9., 25.])
        else:
            ax1.set_xlim([9., 24.])
        ax1.set_ylim([0., 1.1])
        ax1.set_ylabel('f1/2_score',fontsize=16)
        ax1.set_title("f1/2-score-mag",fontsize=16)
        ax1.set_xlabel('Magnitude', fontsize=16)
        plt.savefig('/home/lab30202/lq/ai_future/low_exporsure_emsemble/fpn_resnet_detnet_v2/validation/match_visualization/f1-score_{}.jpg'.format(threhold))
        plt.close()
print("completed!!!")





































