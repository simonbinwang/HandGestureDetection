num_classes = 6  # hand gesture number + face

# training setting
inp_size = (192, 144) 
out_size = (12, 9)


# inference setting
infer_inp_size = (320, 240)
infer_out_size = (20, 15)


num_anchors = 1
anchors = [
    [2.5, 2.5],
]

# loss params
object_scale = 5.
noobject_scale = 1.
class_scale = 1
coord_scale = 1.
iou_thresh = 0.6

# show
class_names = ['five', 'l', 'one', 'seeyou', 'zero', 'face']
colors = [(255,0,0), (0,255,0), (0,0,255), (50, 100, 150), (150, 100, 50), (0,0,0)]
