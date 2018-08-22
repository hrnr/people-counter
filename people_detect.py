#!/usr/bin/env python3


"""
People detector based on MobileNet-SSD.

This is used by people_counter.py
"""

import numpy as np
import cv2 as cv

from common import Rect

# input dimensions for MobileNet-SSD object detection network
# https://github.com/chuanqi305/MobileNet-SSD/
inWidth = 300
inHeight = 300
WHRatio = inWidth / float(inHeight)
inScaleFactor = 0.007843
meanVal = 127.5

# classes that network detects
classNames = ['background',
'aeroplane', 'bicycle', 'bird', 'boat',
'bottle', 'bus', 'car', 'cat', 'chair',
'cow', 'diningtable', 'dog', 'horse',
'motorbike', 'person', 'pottedplant',
'sheep', 'sofa', 'train', 'tvmonitor']

class PeopleDetector:
    def __init__(self, proto, model, confidence):
        # load net for the OpenCV from caffe model and weights
        self.net = cv.dnn.readNetFromCaffe(proto, model)
        # confidence threshold for detections
        self.confidence = confidence
        # detected people
        self.people = []
        # detected other classes than people
        self.other_objects = []

    def update(self, image):
        """
        Detects people and other object in the frame
        """
        # remove previous detections
        self.people = []
        self.other_objects = []
        frame = image.copy()

        # normalize the image for the network input
        blob = cv.dnn.blobFromImage(frame, inScaleFactor, (inWidth, inHeight), (meanVal, meanVal, meanVal), False, False)
        # run the frame though the network
        self.net.setInput(blob)
        detections = self.net.forward()

        # rows and columns of the image
        cols = frame.shape[1]
        rows = frame.shape[0]

        # get detections with confidence higher than threshold
        for i in range(detections.shape[2]):
            # confidence for this detection
            confidence = detections[0, 0, i, 2]
            # type of detected object
            class_id = int(detections[0, 0, i, 1])
            if confidence < self.confidence:
                continue
            if class_id >= len(classNames):
                # unknown object
                continue

            # bounding box coordinates
            xLeftBottom = int(detections[0, 0, i, 3] * cols)
            yLeftBottom = int(detections[0, 0, i, 4] * rows)
            xRightTop   = int(detections[0, 0, i, 5] * cols)
            yRightTop   = int(detections[0, 0, i, 6] * rows)
            # create rectangle for bounding box
            roi = Rect((xLeftBottom, yLeftBottom), (xRightTop, yRightTop))

            # save people and other object separately
            if classNames[class_id] == 'person':
                # save roi
                self.people.append(roi)
            else:
                # save just class name for other objects
                self.other_objects.append(classNames[class_id])

    def visualise(self, frame):
        """
        Visualises detections in current frame
        """
        # draw rectangle for each person
        for roi in self.people:
            cv.rectangle(frame, roi.tl(), roi.br(), (0, 255, 0))

        cv.imshow("detections", frame)

def main():
    """
    main method to run this as indendent script
    """
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0

    print(__doc__)
    detector = PeopleDetector(sys.argv[2], sys.argv[3], 0.5)
    cam = cv.VideoCapture(video_src)
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        detector.update(frame)
        detector.visualise(frame)
        if cv.waitKey(1) >= 0:
            break
    cam.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
