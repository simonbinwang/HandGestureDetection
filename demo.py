import cv2
import caffe
import sys
import time
from easydict import EasyDict
from collections import deque
import copy


from utils import *


# for fast detection 
cfg = EasyDict()
cfg.crop=(0,0)
cfg.infer_inp_size = (320, 240)
cfg.infer_out_size = (20, 15)
cfg.crop_size = (12 * 16, 9 * 16)
cfg.crop_out_size = (12, 9)
cfg.anchors = [
    [2.5, 2.5]
]
cfg.num_anchors = 1
cfg.num_classes = 6
cfg.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
              (50, 100, 150), (150, 100, 50), (0, 0, 0)]
cfg.class_names = ['five', 'l', 'one', 'seeyou', 'zero', 'face']
cfg_backup = copy.deepcopy(cfg) 


if __name__ == '__main__':
    assert len(sys.argv) >= 4  # must give prototxt and caffemodel
    model_def = sys.argv[1]
    model_weights = sys.argv[2]
    output_name = sys.argv[3]
    thresh = 0.5 if len(sys.argv) < 5 else float(sys.argv[4])
    print thresh

    # init model
    yolohandnet = caffe.Net(model_def,      # defines the structure of the model
                            model_weights,  # contains the trained weights
                            caffe.TEST)     # use test mode (e.g., don't perform dropout)


    # create transformer for the input called 'data'
    mu = np.asarray([0.5, 0.5, 0.5]) * 255
    print 'mean-subtracted values:', zip('RGB', mu)
    print 'mdoel input size: {}'.format(cfg.infer_inp_size + [3])
    transformer = caffe.io.Transformer(
        {'data': yolohandnet.blobs['data'].data.shape})  # target size
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', mu)
    transformer.set_input_scale('data', 1 / 255. * 1 / 0.25)

    # for click detection
    fmap = deque(maxlen=20)

    # for fps avg
    fpss = deque(maxlen=10)

    # video input
    cam = cv2.VideoCapture('/dev/video1')
    while True:
        # get frame
        ret, frame = cam.read()
        if ret == False:
            break

        # start inference
        since = time.time()

        # pre process
        cimg = cv2.resize(frame, tuple(cfg_backup.infer_inp_size))
        img = cv2.cvtColor(cimg, cv2.COLOR_BGR2RGB)
        transformed_image = transformer.preprocess('data', img)

        ## crop preprocess
        # print cfg.inp_size
        transformed_image = transformed_image[:, cfg.crop[1]: cfg.crop[1] + cfg.infer_inp_size[1], cfg.crop[0]: cfg.crop[0]+cfg.infer_inp_size[0]]
        # print cfg

        # forward
        yolohandnet.blobs['data'].reshape(1,        # batch size
                                            # 3-channel (BGR) images
                                            3,
                                            cfg.infer_inp_size[1], cfg.infer_inp_size[0])
        yolohandnet.blobs['data'].data[...] = transformed_image
        bbox_pred, iou_pred, prob_pred = net_postprocess(
            yolohandnet.forward()[output_name], cfg)
        # postprocess
        bboxes, scores, cls_inds = postprocess(
            bbox_pred, iou_pred, prob_pred, cfg, thresh=thresh)

        ## crop post process
        crop_postprocess(bboxes, cfg, cfg_backup)

        # click detection
        click_detection(frame, fmap, bboxes, cls_inds, cfg_backup)

        # end inference
        now = time.time()

        fpss.append(1 / (now - since))
        fps = np.mean(np.array(fpss))
        ## draw on origin frame
        frame = my_draw_detection(frame,
                                  bboxes, scores, cls_inds,
                                  cfg,
                                  scale=1.0 * frame.shape[0] / img.shape[0],
                                  thr=0,
                                  fps=fps)

        cv2.imshow('', frame)
        key = cv2.waitKey(1)

        if key is ord('q'):
            break
        
        next_preprocess(bboxes, cfg, cfg_backup)

    cam.release()
