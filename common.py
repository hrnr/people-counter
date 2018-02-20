'''
Common utilities used though this project
'''

import cv2 as cv

def draw_str(dst, target, s):
    """
    Draws text with shadow
    """
    x, y = target
    cv.putText(dst, s, (x+1, y+1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)

# some nice colors for visualisation
color_palette = [
    (0,0,0),
    (230,159,0),
    (86,180,223),
    (0,158,115),
    (240,228,66),
    (0,114,178),
    (213,94,0),
    (204,121,167),
]
def randColor(n):
    """
    Pick color from color palette based on n
    """
    return color_palette[n % len(color_palette)]

class Rect:
    """
    2D rectangle. reimplements Rect from OpenCV, which is not exported to Python bindings
    """
    x = 0
    y = 0
    width = 0
    height = 0

    def __init__(self, pt1, pt2):
        """
        Initializes Rectangle using 2 points
        """
        self.x = min(pt1[0], pt2[0]);
        self.y = min(pt1[1], pt2[1]);
        self.width = max(pt1[0], pt2[0]) - self.x;
        self.height = max(pt1[1], pt2[1]) - self.y;

    def tl(self):
        """
        Returns top-left corner
        """
        return (self.x, self.y)

    def br(self):
        """
        Returns bottom-right corner
        """
        return (self.x + self.width, self.y + self.height)

    def area(self):
        """
        Returns area of the rectangle
        """
        return self.width * self.height

    def contains(self, pt):
        """
        Returns true when rectangle contains given Point pt
        """
        return self.x <= pt[0] and pt[0] < self.x + self.width and self.y <= pt[1] and pt[1] < self.y + self.height

