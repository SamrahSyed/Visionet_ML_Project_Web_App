# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 23:41:27 2021

@author: samrah.asif
"""


from matplotlib import pyplot as plt
import gluoncv
from gluoncv import model_zoo, data, utils
from collections import Counter
import cv2
import gluoncv as gcv
from gluoncv.utils import try_import_cv2
cv2 = try_import_cv2()
import mxnet as mx


def FASTER_R_CNN(image):
    net = model_zoo.get_model('faster_rcnn_resnet50_v1b_voc', pretrained=True)
    x, orig_img = data.transforms.presets.rcnn.load_test(image)
    box_ids, scores, bboxes = net(x)
    fig, ax = plt.subplots(figsize=(15,15))
    ax = utils.viz.plot_bbox(orig_img, bboxes[0], scores[0], box_ids[0], class_names=net.classes, ax=ax)

    #return plt.show()
    plt.savefig(pic)
    return

#FASTER_R_CNN('static/playing_celo.jpeg')

def YOLO(image):
    net = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True)
    x, img = data.transforms.presets.yolo.load_test(image, short=512)
    class_IDs, scores, bounding_boxs = net(x)
    fig, ax = plt.subplots(figsize=(15,15))
    ax = utils.viz.plot_bbox(img, bounding_boxs[0], scores[0],
                         class_IDs[0], class_names=net.classes, ax=ax)
    #return plt.show()
    plt.savefig(pic)
    return
#YOLO('static/playing_celo.jpeg')

def SSD(image):
    net = model_zoo.get_model('ssd_512_resnet50_v1_voc', pretrained=True)
    x, img = data.transforms.presets.ssd.load_test(image, short=512)
    class_IDs, scores, bounding_boxes = net(x)
    fig, ax = plt.subplots(figsize=(15,15))
    ax = utils.viz.plot_bbox(img, bounding_boxes[0], scores[0],
                         class_IDs[0], class_names=net.classes, ax=ax)
    #return plt.show()
    plt.savefig(pic)
    return
#SSD('static/playing_celo.jpeg')

def CenterNet(pic):
    net = model_zoo.get_model('center_net_resnet18_v1b_voc', pretrained=True)
    x, img = data.transforms.presets.center_net.load_test(pic, short=512)
    class_IDs, scores, bounding_boxs = net(x)
    fig, ax = plt.subplots(figsize=(15,15))
    ax = utils.viz.plot_bbox(img, bounding_boxs[0], scores[0],
                         class_IDs[0], class_names=net.classes, ax=ax)
    #return plt.show()
    plt.savefig(pic)
    return
#CenterNet('static/playing_celo.jpeg')

def MOB_NET (video):
    net = gcv.model_zoo.get_model('ssd_512_mobilenet1.0_voc', pretrained=True)
    cap = cv2.VideoCapture(video)
    axes = None
    NUM_FRAMES = 200 # you can change this
    for i in range(NUM_FRAMES):
        ret, frame = cap.read()
        frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')
        rgb_nd, frame = gcv.data.transforms.presets.ssd.transform_test(frame, short=512, max_size=700)
        class_IDs, scores, bounding_boxes = net(rgb_nd)

        img = gcv.utils.viz.cv_plot_bbox(frame, bounding_boxes[0], scores[0], class_IDs[0], class_names=net.classes)
        gcv.utils.viz.cv_plot_image(img)
        cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(2)

    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # if we are supposed to be writing a video to disk, initialize
    # the writer
    writer=None
    output="/static/uploads/obd.mp4"
    if output is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(output, fourcc, 30,
            (W, H), True)
        print(output)
    
MOB_NET('static/v_Basketball_g01_c01.mp4')