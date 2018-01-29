#!/usr/bin/env python

# python 3 compatibility
from __future__ import absolute_import, division, print_function
from builtins import *
from future.builtins.disabled import *

import argparse
import random
import time
from collections import namedtuple

import numpy as np
import cv2 as cv

from people_detect import PeopleDetector
from lucas_kanade import LucasKanadeTracker
from common import draw_str

def nearBorder(rect, rows, cols, border_width):
    border_rect = rect.tl()[0] < border_width or rect.tl()[0] < border_width
    border_rect |= rect.br()[0] > rows + border_width or rect.br()[1] > border_width + cols
    return border_rect

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
# pick random color based on n
def randColor(n):
    return color_palette[n % len(color_palette)]

def matchRoisFromFlow(old_roi, new_roi, tracks, step):
    matched_tracks = 0
    for track in tracks:
        # we need at least tracks that are in the this frame and frame
        # with previous detection
        if len(track) < step + 1:
            continue
        if new_roi.contains(track[-1]) and old_roi.contains(track[-1 - step]):
            matched_tracks += 1
    return matched_tracks

def main():
    parser = argparse.ArgumentParser(description='Counting people in videos')
    parser.add_argument("--video", help="path to video file. If empty, camera's stream will be used")
    parser.add_argument("--proto", default="MobileNetSSD_deploy.prototxt", help='Path to text network file: MobileNetSSD_deploy.prototxt')
    parser.add_argument("--model", default="MobileNetSSD_deploy.caffemodel", help='Path to weights: MobileNetSSD_deploy.caffemodel')
    parser.add_argument("--confidence", default=0.6, type=float, help="confidence threshold to filter out weak detections")
    args = parser.parse_args()

    PeopleCounter(args).run()

RectStamped = namedtuple('RectStamped', ['roi', 'stamp'])

class PeopleCounter:
    def __init__(self, args):
        self.min_tracks_for_match = 7
        self.detect_interval = 8
        self.finish_tracking_after = self.detect_interval + 3
        self.min_track_length = 4

        self.detector = PeopleDetector(args.proto, args.model, args.confidence)
        self.tracker = LucasKanadeTracker(50)
        self.people = []
        self.count_passed = 0
        self.frame_idx = 0
        self.total_time = 1
        if args.video:
            self.cap = cv.VideoCapture(args.video)
        else:
            self.cap = cv.VideoCapture(0)

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.vis = frame.copy()

            # measure time
            start = time.clock()

            # optical flow update - fast
            self.tracker.update(frame)

            if self.frame_idx % self.detect_interval == 0:
                # detect new keypoints
                self.tracker.detect(frame)
                # detect people
                self.detector.update(frame)
                # count people
                self.count()

                self.tracker.visualise(frame)
                self.visualise(frame)
            # measure time
            stop = time.clock()
            self.total_time += stop - start
            self.frame_idx += 1
            # wait for visualisation
            if cv.waitKey(1) == 27:
                break

        self.cap.release()
        cv.destroyAllWindows()
        print('passed people: ', self.count_passed)

    def count(self):
        self._add_new_people()
        self._count_finished_tracks()

    def _add_new_people(self):
        # find if we have new person in the image
        for new_person in self.detector.people:
            # ignore detections that are close to border, since there we might not be
            # to track people reliably (depends on camera setup)
            # if nearBorder(new_person, self.vis.shape[0], self.vis.shape[1], self.border_width):
            #     continue
            # try to match detected persons to the new ones
            matches = np.zeros(len(self.people))
            for i,person_track in enumerate(self.people):
                # count tracks that match
                matches[i] = matchRoisFromFlow(person_track[-1].roi, new_person,
                    self.tracker.tracks, self.detect_interval)
            if self.people:
                # choose person with maximum matches
                max_ind = np.argmax(matches)
                matched_tracks = matches[max_ind]
            else:
                matched_tracks = 0

            if matched_tracks > self.min_tracks_for_match:
                # we matched an old person
                # add to detecction to person's track
                self.people[max_ind].append(RectStamped(new_person, self.frame_idx))
                # print('old match: ', matched_tracks)
            else:
                # we haven't found a matching person, lets add new one
                self.people.append([RectStamped(new_person, self.frame_idx)])
                # print('new_person', new_person)
                # self.visualise()
                # if cv.waitKey(-1):
                #     pass

    def _count_finished_tracks(self):
        for person_track in self.people:
            if self.frame_idx - person_track[-1].stamp < self.finish_tracking_after:
                # this track is still new
                continue
            # track is too old
            self.people.remove(person_track)
            # remove outliers - too short tracks are probably just noise
            if len(person_track) < self.min_track_length:
                continue
            # increase counters
            self.count_passed += 1

    def visualise(self, frame):
        draw_str(frame, (20, 40), 'passed people: %d' % self.count_passed)
        draw_str(frame, (20, 60), 'fps: %d' % (self.frame_idx / self.total_time))
        for track in self.people:
            color = randColor(hash(track[0]))
            for roi, _ in track:
                cv.rectangle(frame, roi.tl(), roi.br(), color)

        cv.imshow("counter", frame)

if __name__ == "__main__":
    main()
