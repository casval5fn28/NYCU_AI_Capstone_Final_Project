from time import sleep
import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
from glob import glob
#from PIL import image
frame_count = 0      # used in mainloop, extract imgs and then to drawPred( called by post process)
frame_count_out=0    # used in post process loop, to get the no of specified class value.
# initialize the parameters
confThreshold = 0.5  # confidence threshold
nmsThreshold = 0.4   # non-maximum suppression threshold
inpWidth = 416       # width of network's input img
inpHeight = 416      # height of network's input img


# load names of classes
classesFile = "obj.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# give the configuration and weight files for model and load the network.
modelConfiguration = "yolov3-obj.cfg"
modelWeights = "yolov3-obj_2400.weights"

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# get names of the output layers
def getOutputsNames(net):
    # names of all layers in the network
    layersNames = net.getLayerNames()
    # names of output layers, i.e. with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom, frame):
    global frame_count
    #cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    label = '%.2f' % conf
    # get label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    # display label at top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])

    label_name,label_conf = label.split(':')# split into class&confidance. compare it with person
    if label_name == 'Helmet':
        #cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
        #cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
        frame_count+=1
    #print(frame_count)
    if(frame_count> 0):
        return frame_count

# remove bounding boxes with low confidence by non-maxima suppression
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    frame_count_out=0
    classIds = []
    confidences = []
    boxes = []
    # scan all bounding boxes output, only keep ones with high confidence scores
    # assign a box's class label as class with highest score
    classIds = []          
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                #print(classIds)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # perform non maximum suppression
    # eliminate redundant overlapping boxes with lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    count_person=0 # counting classes in this loop.
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        #frame_count_out = drawPred(classIds[i], confidences[i], left, top, left + width, top + height, frame)
        #increase test counter till loop end

        #check the class and see if it's a person
        my_class='Helmet'
        unknown_class = classes[classId]

        if my_class == unknown_class:
            count_person += 1
    #if(frame_count_out > 0):
        #print(frame_count_out)

    if count_person >= 1:
        return 1
    else:
        return 0    

# process inputs
winName = 'Deep learning object detection in OpenCV'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

def detect(frame):
    #frame = cv.imread(fn)

    # create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
    # set input to the network
    net.setInput(blob)

    # run the forward pass to get output of output layers
    outs = net.forward(getOutputsNames(net))

    # overall time for inference(t) and timings for each layer(in layersTimes)
    t, _ = net.getPerfProfile()
    #print(t)
    #label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    #print(label)
    #cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    #print(label)
    # remove bounding boxes with low confidence
    k = postprocess(frame, outs)
    if k:
        return 1
    else:
        return 0
 
 
 
