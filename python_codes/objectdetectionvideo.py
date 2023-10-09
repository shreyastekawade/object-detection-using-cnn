######## Video Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/16/18
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier and uses it to perform object detection on a video.
# It draws boxes, scores, and labels around the objects of interest in each
# frame of the video.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import time

def convertToCentroid(vboxes, w, h):
    centroids = []
    for vbox in vboxes:
        res = [(vbox[1]+vbox[3])/2, (vbox[0]+vbox[2])/2]
        centroids.append(res)
    return centroids

def getDistances(vbox1, vbox2):
    dist1 = abs(np.array(vbox1) - np.array(vbox2))
    dist2 = abs(np.array(vbox1) - np.array([vbox2[2], vbox2[3], vbox2[0], vbox2[1]]))
    print(vbox1, vbox2, dist1)
    print(vbox1, [vbox2[2], vbox2[3], vbox2[0], vbox2[1]], dist2)
    return [dist1, dist2]


from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'trained-inference-graphs1/frozen_inference_graph'
VIDEO_NAME = 'test.mp4'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training/labelmap.pbtxt')

# Path to video
PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 3

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and put tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Open video file
video = cv2.VideoCapture(PATH_TO_VIDEO)
 #IMAGE_NAME=PATH_TO_VIDEO.split("/")[-1]

while(video.isOpened()):

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = video.read()  
    w=1920
    h=1080


    frame_expanded = np.expand_dims(frame, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})


    sc = scores.tolist()[0]
    bx = boxes.tolist()[0]
    cls= classes.tolist()[0]
    score_ind = [sc.index(score) for score in sc if score >= 0.72]
    #print(score_ind)
    vis_boxes = [bx[var] for var in score_ind]
    #print(vis_boxes)
    vis_classes = [cls[var] for var in score_ind]

    centroid_array = convertToCentroid(vis_boxes, w, h)
    #print(centroid_array)

    max_threshX = 150/w
    max_threshY = 150/h

    for vb1 in vis_boxes:
        for vb2 in vis_boxes:
            i1=vis_boxes.index(vb1)
            i2=vis_boxes.index(vb2)

            if i1 == i2:
                continue
            if vis_classes[i1] == vis_classes[i2]:
                continue
            print(vis_boxes.index(vb1), '----', vis_boxes.index(vb2))
            [dist1, dist2] = getDistances(vb1, vb2)
            #print(dist1, '\t', dist2)
            bool1 = dist1 < np.array([max_threshY, max_threshX, max_threshY, max_threshX])
            bool2 = dist2 < np.array([max_threshY, max_threshX, max_threshY, max_threshX])
            #]print(bool1.astype(int).sum(), '\t', bool2.astype(int).sum())
            if bool1.astype(int).sum() or bool2.astype(int).sum():
                print("Run away")
        
    #np.savetxt('records/boxes/array_boxes1'+IMAGE_NAME+'.csv', np.squeeze(boxes), delimiter=',', fmt='%2.4f')
    #np.savetxt('records/scores/array_scores1'+IMAGE_NAME+'.csv', np.squeeze(scores), delimiter=',', fmt='%2.4f')
    #np.savetxt('records/classes/array_classes1'+IMAGE_NAME+'.csv', np.squeeze(classes).astype(np.int32), delimiter=',', fmt='%2d')


    # Draw the results of the detection (aka 'visulaize the results')
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.60)

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
cv2.destroyAllWindows()
