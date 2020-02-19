import cv2
import numpy as np
import os
import glob

def vis_detections(im,dets):
    """Visual debugging of detections.
    第一个参数：img是原图
    第二个参数：（x，y）是矩阵的左上点坐标
    第三个参数：（x+w，y+h）是矩阵的右下点坐标
    第四个参数：（0,255,0）是画线对应的rgb颜色
    第五个参数：2是所画的线的宽度
    """
    # for i in range(np.minimum(10, dets.shape[0])):
    for i in range(dets.shape[0]):
        # bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        if 14 <= dets[i][3] <= 24:
            padding = 5
            x1 = np.around(dets[i][1] - padding)
            x2 = np.around(dets[i][1] + padding)
            y1 = np.around(dets[i][2] - padding)
            y2 = np.around(dets[i][2] + padding)
        elif 11 <= dets[i][3] <= 14:
            padding = 7.5
            x1 = np.around(dets[i][1] - padding)
            x2 = np.around(dets[i][1] + padding)
            y1 = np.around(dets[i][2] - padding)
            y2 = np.around(dets[i][2] + padding)
        elif 10 <= dets[i][3] <= 11:
            padding = 10
            x1 = np.around(dets[i][1] - padding)
            x2 = np.around(dets[i][1] + padding)
            y1 = np.around(dets[i][2] - padding)
            y2 = np.around(dets[i][2] + padding)
        elif 9 <= dets[i][3] <= 10:
            padding = 12.5
            x1 = np.around(dets[i][1] - padding)
            x2 = np.around(dets[i][1] + padding)
            y1 = np.around(dets[i][2] - padding)
            y2 = np.around(dets[i][2] + padding)
        x1 = max(int(x1), 0)
        x2 = max(int(x2), 0)
        y1 = max(int(y1), 0)
        y2 = max(int(y2), 0)
        bbox = (x1,y1,x2,y2)
        if dets[i][0] == 100:
            cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 204, 0), 1)   ### green
                # cv2.putText(im, '%s: %.3f' % (class_name, score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                #                 #             1.0, (0, 204, 0), thickness=1)
        else:
            cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 0, 255), 1)   ### red
                # cv2.putText(im, '%s: %.3f' % (class_name, score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                #             1.0, (0, 0, 255), thickness=1)
    return im


save_path = "/home/lab30202/lq/ai_future/low_exporsure_emsemble/fpn_resnet_detnet_v2/validation/check_GT"

list_list = glob.glob(os.path.join(save_path,"*.list"))
for single_list in list_list:
    list_img = np.loadtxt(single_list)
    image_index = single_list.split("/")[-1].split(".list")[0]
    image_name = os.path.join(save_path,image_index+".jpg")
    img = cv2.imread(image_name)
    im_jpg = cv2.flip(img, 0)
    bb_img = vis_detections(im_jpg, list_img)
    result_path = os.path.join(save_path, image_index+"_gt.jpg")
    cv2.imwrite(result_path, bb_img)




