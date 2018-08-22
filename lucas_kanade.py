#!/usr/bin/env python3

"""
Lucas-Kanade tracker
====================

Lucas-Kanade sparse optical flow based tracker. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.

This is used by people_detect.py
"""

import numpy as np
import cv2 as cv
from common import draw_str
from time import clock
import random

class LucasKanadeTracker:
    def __init__(self, track_len):
        # how much long tracks to keep
        self.track_len = track_len
        # list of lists of tracked points
        self.tracks = []
        # Lucas-Kanade parameters
        self.lk_params = dict(winSize = (15, 15), maxLevel = 2,
                              criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
        # parameters for goodFeaturesToTrack
        self.feature_params = dict(maxCorners = 600, qualityLevel = 0.1, minDistance = 7, blockSize = 7)

    def update(self, image):
        """
        Runs fast Lucas-Kanade tracking without re-detectiing the feature points
        """
        # convert to grayscale
        frame_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        if len(self.tracks) > 0:
            # image to compute flow between
            img0, img1 = self.prev_gray, frame_gray
            # last known points positions (from previous frame)
            p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
            # forward flow
            p1, st, err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **self.lk_params)
            # backwards flow
            p0r, st, err = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **self.lk_params)
            # error between forward flow and backwards flow
            d = abs(p0-p0r).reshape(-1, 2).max(-1)
            # mask for features that have matching forward and backwards flow
            good = d < 1
            new_tracks = []
            for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                if not good_flag:
                    # skip unreliable points
                    continue
                # add new point to the track
                tr.append((x, y))
                if len(tr) > self.track_len:
                    # remove oldest point if the track is too long
                    del tr[0]
                new_tracks.append(tr)
            # save the new tracks
            self.tracks = new_tracks
        # save the current image
        self.prev_gray = frame_gray

    def detect(self, image):
        """
        Detects new feature points
        """
        # convert to gray
        frame_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # initialize mask
        mask = np.zeros_like(frame_gray)
        mask[:] = 255
        # create mask to supress detecting features around points we already track
        for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
            cv.circle(mask, (x, y), 5, 0, -1)
        # detect feature points using minimal eigenvalue of gradient matrices method
        p = cv.goodFeaturesToTrack(frame_gray, mask = mask, **self.feature_params)
        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                # add new feature point for tracking
                self.tracks.append([(x, y)])

    def visualise(self, frame):
        """
        Draw visualised tracks to frame
        """
        # draw point for each feature point
        for track in self.tracks:
            cv.circle(frame, track[-1], 2, (0, 255, 0), -1)
        # draw path for each track
        cv.polylines(frame, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
        # print how many tracks we have altogether
        draw_str(frame, (20, 20), 'track count: %d' % len(self.tracks))

def main():
    """
    This can be used to run only Lucas-Kande tracker independently
    """
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0

    print(__doc__)
    tracker = LucasKanadeTracker(100)
    cam = cv.VideoCapture(video_src)
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        tracker.update(frame)
        tracker.detect(frame)
        tracker.visualise(frame)
        cv.imshow('lk_track', frame)
        if cv.waitKey(1) >= 0:
            break
    cam.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
