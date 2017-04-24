# CNN_pedestrian_tracking
This is the code in MATLAB for the works related to the paper "Real-time detection and tracking of pedestrians in CCTV images using a deep convolutional neural network"
The dataset used is available at http://www.robots.ox.ac.uk/ActiveVision/Research/Projects/2009bbenfold_headpose/project.html#datasets/. 
First 250 frames of the data is used for evaluation
The detections are done using Faster-RCNN on the above dataset
Attached .mat are the files containing the coordinates of the bounding boxes as output of the Faster-RCNN (People category), activitions of the detections from the last layer of the CNN.
