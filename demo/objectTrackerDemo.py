#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 09:48:13 2021

@author: alexander
"""

import torchvision.transforms as T
from torchvision.io import read_image
from pathlib import Path
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes
from torchvision import transforms
from torch.autograd import Variable as V
import time
from CentroidTracker import CentroidTracker
import time
#from imutils.video import VideoStream

#import imutils

def draw_bounding_boxes2(img, prediction, rects):
    
    color = (255,0,0)
    
    boxes = torch.as_tensor(prediction[0]['boxes'].cpu().numpy(), dtype=torch.float)
    labels_int = (prediction[0]['labels'].cpu().numpy()).tolist()
    scores = (prediction[0]['scores'].cpu().numpy()).tolist()
    labels_str=[]
    for i,coord in enumerate(boxes.tolist()):
    
        if(scores[i] > 0.5):
            x1 = int(coord[0])
            y1 = int(coord[1])
            x2 = int(coord[2])
            y2 = int(coord[3])
            
            box = [x1, y1, x2, y2]
            rects.append(box)
            cv2.rectangle(img, (x1,y1), (x2, y2), color, 2)
            
            
            if(labels_int[i]==1):
                labels_str.append("cola: "+str(round(scores[i],2)))
                text = "cola"
            else:
                text = "beer"
                labels_str.append("beer"+str(round(scores[i],2)))
                    
            # Add label with score
            cv2.putText(img, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255),1)
            cv2.putText(img, str(int(scores[i]*100)), (x1+50, y1), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255),1)

    
    return img, rects

def get_instance_segmentation_model(num_classes):
    
    # load a pre-trained model for classification and return
    # only the features
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    #backbone = torchvision.models.mobilenet_v3_small(pretrained=True)
    # FasterRCNN needs to know the number of
    # output channels in a backbone. For mobilenet_v2, it's 1280
    # so we need to add it here
    backbone.out_channels = 1280

    # let's make the RPN generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios 
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))

    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be [0]. More generally, the backbone should return an
    # OrderedDict[Tensor], and in featmap_names you can choose which
    # feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)

    # put the pieces together inside a FasterRCNN model
    model = FasterRCNN(backbone,
                       num_classes=3,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)
    return model
    '''
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes = 3  # 1 class (person) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    return model
    '''
    

if __name__ == "__main__": 
    try:
        ct = CentroidTracker()

        modeName = "epoch_best_mobilenet-10.pt"
        videFile = "video2.avi"
        num_classes = 3
        #cap = cv2.VideoCapture(-1)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    
        #cap = cv2.VideoCapture(0)
        cap = cv2.VideoCapture(videFile)
        time.sleep(2)
    
        tensor = transforms.ToTensor()
        img_transforms = T.Compose([T.ToPILImage(),T.ToTensor()])
        
        model = get_instance_segmentation_model(num_classes)
        model.load_state_dict(torch.load(modeName))
        model.to(device)
        model.eval()
        
        tensor = transforms.ToTensor()
        rectangles_list = []
        rectangles_list2 = []
        frame_number = 0
        tic = 0
        toc = 0
        while(True):
            

            ret, frame = cap.read()
            if not ret: 
                break
            
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            img2 = tensor(img)
            
            with torch.no_grad():
                prediction = model([img2.to(device)])
                
            rectangles_list = []            
            boxes = torch.as_tensor(prediction[0]['boxes'].cpu().numpy(), dtype=torch.float)
            scores = (prediction[0]['scores'].cpu().numpy()).tolist()
            for i,coord in enumerate(boxes.tolist()):
    
                if(scores[i] > 0.95):
                    x1 = int(coord[0])
                    y1 = int(coord[1])
                    x2 = int(coord[2])
                    y2 = int(coord[3])
            
                    box = [x1, y1, x2, y2]
                    rectangles_list.append(box)
          
            drawn_boxes, rectangles_list2 = draw_bounding_boxes2(frame, prediction, rectangles_list2)
            #
            objects = ct.update(rectangles_list)
            # loop over the tracked objects
            for (objectID, centroid) in objects.items():
              		# draw both the ID of the object and the centroid of the
              		# object on the output frame0
                #print(objectID)
                #print(len(rectangles_list))
                text = "ID {}".format(objectID)
                cv2.putText(drawn_boxes, text, (centroid[0] - 10, centroid[1] - 10),
                  		cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.circle(drawn_boxes, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                
            cv2.imshow("test",drawn_boxes)

            frame_number = frame_number + 1
            tic = time.time()
            fps = 1/(tic-toc)
            toc = tic
            cv2.putText(drawn_boxes, "Fps:"+ str(fps), (20, 20),
                  		cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow("test",drawn_boxes)
            if(frame_number%5 == 0):
                cv2.imwrite("detectionTest_images/fig_"+str(frame_number)+".png", drawn_boxes) 
                #print(frame_number)
            if cv2.waitKey(10) & 0xFF == 27:
                cv2.destroyAllWindows()
                cap.release()
                break
    except KeyboardInterrupt:       
        cv2.destroyAllWindows()
        cap.release()




