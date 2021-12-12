#!/usr/bin/python3
# 2019/01/24 09:05
# 2019/01/24 10:25

import gluoncv as gcv
import mxnet as mx
import cv2
import numpy as np
# https://github.com/pjreddie/darknet/blob/master/data/dog.jpg

## (1) Create network 
net = gcv.model_zoo.get_model('ssd_512_mobilenet1.0_voc', pretrained=True)

## (2) Read the image and preprocess 
img = cv2.imread("output_205.jpg")
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

xrgb = mx.nd.array(rgb).astype('uint8')
rgb_nd, xrgb = gcv.data.transforms.presets.ssd.transform_test(xrgb, short=512, max_size=700)

## (3) Interface 
class_IDs, scores, bounding_boxes = net(rgb_nd)

## (4) Display 
for i in range(len(scores[0])):
    #print(class_IDs.reshape(-1))
    #print(scores.reshape(-1))
    cid = int(class_IDs[0][i].asnumpy())
    cname = net.classes[cid]
    score = float(scores[0][i].asnumpy())
    if score < 0.5:
        break
    x,y,w,h = bbox =  bounding_boxes[0][i].astype(int).asnumpy()
    print(cid, score, bbox)
    tag = "{}; {:.4f}".format(cname, score)
    cv2.rectangle(img, (x,y), (w, h), (0, 255, 0), 2)
    cv2.putText(img, tag, (x, y-20),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)

cv2.imshow("ssd", img);
cv2.waitKey()