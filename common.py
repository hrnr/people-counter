'''
Common utilities used though this project
'''

import numpy as np
import cv2 as cv

def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x+1, y+1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)

# 2d rectangle
class Rect:
    x = 0
    y = 0
    width = 0
    height = 0

    def __init__(self, pt1, pt2):
        self.x = min(pt1[0], pt2[0]);
        self.y = min(pt1[1], pt2[1]);
        self.width = max(pt1[0], pt2[0]) - self.x;
        self.height = max(pt1[1], pt2[1]) - self.y;

    def tl(self):
        return (self.x, self.y)

    def br(self):
        return (self.x + self.width, self.y + self.height)

    def area(self):
        return self.width * self.height

    def contains(self, pt):
        return self.x <= pt[0] and pt[0] < self.x + self.width and self.y <= pt[1] and pt[1] < self.y + self.height

