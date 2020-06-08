import numpy as np
import matplotlib.pyplot as plt
import cv2

import mxnet as mx
from mxnet import gluon, nd
import gluoncv as gcv
from gluoncv import data, utils
from gluon_utils.model_zoo.centernet import get_center_net_transfer

classes = ['pikachu']
net = get_center_net_transfer(
    'center_net_resnet18_v1b_coco', classes, pretrained=True)
net.load_parameters("data/checkpoints/epoch_9.params")

dataset = data.RecordFileDetection('data/pikachu_train.rec')
image, label = dataset[0]
image = cv2.resize(image.asnumpy(), (512, 512))
x, img = data.transforms.presets.center_net.transform_test(
    nd.array(image), short=512)

class_IDs, scores, bounding_boxes = net(x)
ax = utils.viz.plot_bbox(
    img, bounding_boxes[0], scores[0], class_IDs[0], class_names=net.classes)
plt.show()