import numpy as np
import torch

def Giou_np(bbox_p, bbox_g,reduction = "mean"):
    """
    :param bbox_p: predict of bbox(N,4)(x1,y1,x2,y2)
    :param bbox_g: groundtruth of bbox(N,4)(x1,y1,x2,y2)
    :return:
    """
    # for details should go to https://arxiv.org/pdf/1902.09630.pdf
    # ensure predict's bbox form
    batch_size = bbox_p.shape[0]
    bbox_g = bbox_g.view(-1,4)
    bbox_p = bbox_p.view(-1,4)
    bbox_p = bbox_p.cpu().detach().numpy()
    bbox_g = bbox_g.cpu().detach().numpy()
    x1p = np.minimum(bbox_p[:, 0], bbox_p[:, 2]).reshape(-1,1)
    x2p = np.maximum(bbox_p[:, 0], bbox_p[:, 2]).reshape(-1,1)
    y1p = np.minimum(bbox_p[:, 1], bbox_p[:, 3]).reshape(-1,1)
    y2p = np.maximum(bbox_p[:, 1], bbox_p[:, 3]).reshape(-1,1)

    bbox_p = np.concatenate([x1p, y1p, x2p, y2p], axis=1)
    # calc area of Bg
    area_p = (bbox_p[:, 2] - bbox_p[:, 0]) * (bbox_p[:, 3] - bbox_p[:, 1])
    # calc area of Bp
    area_g = (bbox_g[:, 2] - bbox_g[:, 0]) * (bbox_g[:, 3] - bbox_g[:, 1])

    # cal intersection
    x1I = np.maximum(bbox_p[:, 0], bbox_g[:, 0])
    y1I = np.maximum(bbox_p[:, 1], bbox_g[:, 1])
    x2I = np.minimum(bbox_p[:, 2], bbox_g[:, 2])
    y2I = np.minimum(bbox_p[:, 3], bbox_g[:, 3])
    I = np.maximum((y2I - y1I), 0) * np.maximum((x2I - x1I), 0)

    # find enclosing box
    x1C = np.minimum(bbox_p[:, 0], bbox_g[:, 0])
    y1C = np.minimum(bbox_p[:, 1], bbox_g[:, 1])
    x2C = np.maximum(bbox_p[:, 2], bbox_g[:, 2])
    y2C = np.maximum(bbox_p[:, 3], bbox_g[:, 3])

    # calc area of Bc
    area_c = (x2C - x1C) * (y2C - y1C)
    U = area_p + area_g - I
    iou = 1.0 * I / U

    # Giou
    giou = iou - (area_c - U) / area_c

    # loss_iou = 1 - iou loss_giou = 1 - giou
    loss_iou = 1.0 - iou
    loss_giou = 1.0 - giou
    loss_giou = torch.FloatTensor(loss_giou).cuda()
    if reduction == 'mean':
        loss_giou = loss_giou.mean()
    elif reduction == 'sum':
        loss_giou = loss_giou.sum()
    elif reduction == 'none':
        pass
    return giou, loss_iou, loss_giou


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    p = np.array([[21,45,103,172],
                  [34,283,155,406],
                  [202,174,271,255]])
    g = np.array([[59,106,154,230],
                  [71,272,191,419],
                  [257,244,329,351]])
    giou, loss_iou, loss_giou = Giou_np(p, g)
    print("giou:",giou,
          "\nloss_iou:",loss_iou,
          "\nloss_giou:",loss_giou
          )