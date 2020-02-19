import numpy as np
import glob
import os
import codecs
path = "/home/lab30202/lq/ai_future/low_exporsure_emsemble/fpn_resnet_detnet_v2/validation"
list_path = os.path.join(path,"*det.txt")
list = glob.glob(list_path)
threholds = [0.4,0.5,0.6]

for threhold in threholds:
    for step,single_list in enumerate(list):
        f = codecs.open(single_list, mode='r', encoding='utf-8')  # 打开txt文件，以‘utf-8'编码读取
        line = f.readline()  # 以行的形式进行读取文件
        name = single_list.split("/")[-1]
        x_min = []
        x_max = []
        y_min = []
        y_max = []
        score = []
        while line:
            a = line.split()
            x_min.append(a[0:1])
            x_max.append(a[1:2])
            y_min.append(a[2:3])
            y_max.append(a[3:4])
            score.append(a[4:5])
            line = f.readline()
        f.close()
        x_min = np.array(x_min)
        x_min = x_min.reshape(len(x_min), 1)
        x_max = np.array(x_max)
        x_max = x_max.reshape(len(x_max), 1)
        y_min = np.array(y_min)
        y_min = y_min.reshape(len(y_min), 1)
        y_max = np.array(y_max)
        y_max = y_max.reshape(len(y_max), 1)
        score = np.array(score)
        score = score.reshape(len(score),1)
        new_list = np.concatenate([x_min, x_max, y_min, y_max, score], axis=1)
        objs_all_list = []
        for step, value_single_row in enumerate(new_list):
            if threhold <= float(new_list[step][4]) <= 1:  ##  remove nagative coordinate
                objs_all_list.append(value_single_row)
        objs_all = np.array(objs_all_list)
        new_name_index = single_list.split('/')[-1].split('.txt')[0] + '_%s' % threhold
        new_name = os.path.join(path,new_name_index+'.txt')
        np.savetxt(new_name,objs_all,fmt="%s")

print("completed!!!")




