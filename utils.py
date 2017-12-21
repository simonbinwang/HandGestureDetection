from yolo.utils.cython_yolo import yolo_to_bbox
from yolo.utils.nms_wrapper import nms

import numpy as np
import cv2

# import logging
# logging.basicConfig(level=logging.DEBUG)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=len(x.shape) - 1, keepdims=True)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def net_postprocess(y, cfg):
    bsize, c, h, w = y.shape  # c = cfg.num_anchors * (cfg.num_classes + 5)
    y_reshaped = y.transpose(0, 2, 3, 1).reshape(
        bsize, -1, cfg.num_anchors, cfg.num_classes + 5)

    xy_pred = sigmoid(y_reshaped[:, :, :, 0:2])
    wh_pred = np.exp(y_reshaped[:, :, :, 2:4])
    bbox_pred = np.concatenate([xy_pred, wh_pred], 3)

    iou_pred = sigmoid(y_reshaped[:, :, :, 4:5])  # (bsize, wxh, num_a, 1)

    score_pred = y_reshaped[:, :, :, 5:]
    prob_pred = softmax(score_pred)  # (bsize, wxh, num_a, num_cls)

    return bbox_pred, iou_pred, prob_pred

def postprocess(bbox_pred, iou_pred, prob_pred, cfg, thresh):
    """
    bbox_pred: (bsize, HxW, num_anchors, 4) ndarray of float (sig(tx), sig(ty), exp(tw), exp(th))
    iou_pred: (bsize, HxW, num_anchors, 1)
    prob_pred: (bsize, HxW, num_anchors, num_classes)
    """
    num_classes, num_anchors = cfg.num_classes, cfg.num_anchors
    anchors = cfg.anchors
    W, H = cfg.infer_out_size
    # print cfg

    assert bbox_pred.shape[0] == 1, 'postprocess only support one image per batch'

    bbox_pred = yolo_to_bbox(
        np.ascontiguousarray(bbox_pred, dtype=np.float),
        np.ascontiguousarray(anchors, dtype=np.float),
        H, W)
    bbox_pred = np.reshape(bbox_pred, [-1, 4])
    bbox_pred[:, 0::2] *= float(cfg.infer_inp_size[0])
    bbox_pred[:, 1::2] *= float(cfg.infer_inp_size[1])
    bbox_pred = bbox_pred.astype(np.int)

    iou_pred = np.reshape(iou_pred, [-1])
    prob_pred = np.reshape(prob_pred, [-1, num_classes])

    cls_inds = np.argmax(prob_pred, axis=1)
    prob_pred = prob_pred[(np.arange(prob_pred.shape[0]), cls_inds)]
    scores = iou_pred * prob_pred

    ## filter
    # threshold
    keep = np.where(scores >= thresh)

    bbox_pred = bbox_pred[keep]
    scores = scores[keep]
    cls_inds = cls_inds[keep]
    # logging.debug(('gesture_filter', scores.shape, keep))

    # filter out face
    face_idx = cfg.num_classes - 1
    keep = np.where(cls_inds != face_idx)

    bbox_pred = bbox_pred[keep]
    scores = scores[keep]
    cls_inds = cls_inds[keep]
    # logging.debug(('gesture_filter', scores.shape, keep))

    # NMS
    keep = np.zeros(len(bbox_pred), dtype=np.int)
    keep[nms_detections(bbox_pred, scores, 0.3)] = 1
    keep = np.where(keep > 0)

    bbox_pred = bbox_pred[keep]
    scores = scores[keep]
    cls_inds = cls_inds[keep]

    # clip
    bbox_pred = clip_boxes(bbox_pred, cfg.infer_inp_size)

    return bbox_pred, scores, cls_inds


def click_detection(frame, action_map, bbox_pred, cls_inds, cfg_backup):
    W, H = cfg_backup.infer_out_size
    cls_inds_map = - np.ones((H, W))  # bg : -1


    open_idx, close_idx = 0, 4  # five zero

    for i, bbox in enumerate(bbox_pred):
        cx = int(np.mean(bbox[0::2]) / 16)
        cy = int(np.mean(bbox[1::2]) / 16)
        if cls_inds[i] in (open_idx, close_idx):
            cls_inds_map[cy, cx] = cls_inds[i]
        else:
            cls_inds_map[cy, cx] = -2  # other gesture


    action_map.append(cls_inds_map.reshape(-1))

    feature_map = np.array(list(action_map))
    feature_map = feature_map.T

    fmap_max = np.max(feature_map, axis=1)
    fmap_min = np.min(feature_map, axis=1)
    keep = np.intersect1d(np.where(fmap_max > 0), np.where(fmap_min >= -1))

    for i, feature_channel in enumerate(feature_map[keep]):
        open_1 = []
        close_2 = []
        open_3 = []
        for t in range(len(feature_channel)):
            if feature_channel[t] in (open_idx, -1):
                open_1.append(t)
            else:
                break
        for t in range(t, (len(feature_channel))):
            if feature_channel[t] in (close_idx, -1):
                close_2.append(t)
            else:
                break
        for t in range(t, (len(feature_channel))):
            if feature_channel[t] in (open_idx, -1):
                open_3.append(t)
            else:
                break
        if len(open_1 + close_2 + open_3) == len(feature_channel) and len(open_1) > 0 and len(close_2) > 0 and len(open_3) > 0:
            cv2.putText(frame, 'click_{}'.format(keep[i]),
                        # (keep[i] % 24 * 32, keep[i] // 24 * 32), 0, 1, 255, 2)
                        (keep[i] % 20 * 32, keep[i] // 20 * 32), 0, 1, 255, 2)


def clip_boxes(boxes, size):
    """
    Clip boxes to image boundaries.
    """
    if boxes.shape[0] == 0:
        return boxes

    w, h = size

    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], w - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], h - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], w - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], h - 1), 0)
    return boxes


def my_draw_detection(im, bboxes, scores, cls_inds, cfg, scale=1., thr=0.3, fps=0):
    # draw image
    imgcv = np.copy(im)
    h, w, _ = imgcv.shape
    thick = int((h + w) / 300)
    for i, box in enumerate(bboxes):
        if scores[i] < thr:
            continue

        cls_indx = cls_inds[i]

        # my scale
        box = [int(scale * z) for z in box]

        cv2.rectangle(imgcv,
                      (box[0], box[1]), (box[2], box[3]),
                      cfg.colors[cls_indx], thick)
        mess = '%s: %.3f' % (cfg.class_names[cls_indx], scores[i])
        cv2.putText(imgcv, mess, (box[0], box[1] - 12),
                    0, 2e-3 * h, cfg.colors[cls_indx], thick // 3)
    cv2.putText(imgcv, str(fps), (0, h / 10),
                0, 2e-3 * h, 255, thick // 3)

    return imgcv


def nms_detections(pred_boxes, scores, nms_thresh):
    dets = np.hstack((pred_boxes,
                      scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, nms_thresh)
    return keep

def next_preprocess(bbox_pred, cfg, cfg_backup):
    '''
        reduce conv field by gesture detected before
    '''
    Wi, Hi = cfg_backup.infer_inp_size
    if len(bbox_pred) == 1:
        Wc, Hc = cfg_backup.crop_size
        cx = int(np.mean(bbox_pred[0][0::2]))
        cy = int(np.mean(bbox_pred[0][1::2]))
        x0 = min(max(cx - Wc / 2, 0), Wi - Wc)
        y0 = min(max(cy - Hc / 2, 0), Hi - Hc)
        cfg.crop = (x0, y0)
        cfg.infer_inp_size = cfg_backup.crop_size
        cfg.infer_out_size = cfg_backup.crop_out_size
    else:
        cfg.crop = cfg_backup.crop
        cfg.infer_inp_size = cfg_backup.infer_inp_size
        cfg.infer_out_size = cfg_backup.infer_out_size


def crop_postprocess(bboxes, cfg, cfg_backup):
    if cfg.infer_inp_size[0] != cfg_backup.infer_inp_size[0]:
        bboxes[:, 0::2] += cfg.crop[0]
        bboxes[:, 1::2] += cfg.crop[1]

