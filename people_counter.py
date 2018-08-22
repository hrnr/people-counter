#!/usr/bin/env python3

"""
Counts people in videos
"""

import argparse
import random
import time
from collections import namedtuple

# numpy and OpenCV required
import numpy as np
import cv2 as cv

from people_detect import PeopleDetector
from lucas_kanade import LucasKanadeTracker
from common import draw_str, randColor


def matchRoisFromFlow(old_roi, new_roi, tracks, step):
    """
    Matches rois across frames using Lucas-Kanade tracks (tracked points)

    returns: number of matched tracks
    """
    # number of tracks that match
    matched_tracks = 0
    for track in tracks:
        # we need at least tracks that are in the this frame and frame
        # with previous detection
        if len(track) < step + 1:
            continue
        # if the track goes through both rois
        if new_roi.contains(track[-1]) and old_roi.contains(track[-1 - step]):
            matched_tracks += 1
    return matched_tracks

def main():
    # accepted commanline arguments. proto and model must point to valid files
    parser = argparse.ArgumentParser(description='Counting people in videos')
    parser.add_argument("--video", help="path to video file. If empty, camera's stream will be used")
    parser.add_argument("--proto", default="data/MobileNetSSD_deploy.prototxt", help='Path to text network file: MobileNetSSD_deploy.prototxt')
    parser.add_argument("--model", default="data/MobileNetSSD_deploy.caffemodel", help='Path to weights: MobileNetSSD_deploy.caffemodel')
    parser.add_argument("--confidence", default=0.6, type=float, help="confidence threshold to filter out weak detections")
    parser.add_argument("--min_tracks_for_match", default=7, type=int, help="minimum number of points that must match between detection to be considered one track")
    parser.add_argument("--min_track_length", default=4, type=int, help="minimum number of detections in one track to count one person")
    parser.add_argument("--detect_interval", default=8, type=int,
        help="each detect_interval frames people detection and corner detection runs. In between people are tracked only using Lucas-Kanade method.")
    args = parser.parse_args()

    # run main counter
    PeopleCounter(args).run()

# Stamped rectangle. Stores roi of detected person and stamp -
# frame index when the dection occured
RectStamped = namedtuple('RectStamped', ['roi', 'stamp'])

class PeopleCounter:
    def __init__(self, args):
        # see above for parameters descriptions
        self.min_tracks_for_match = args.min_tracks_for_match
        self.detect_interval = args.detect_interval
        self.finish_tracking_after = self.detect_interval * 3
        self.min_track_length = args.min_track_length

        # detector for people detection
        self.detector = PeopleDetector(args.proto, args.model, args.confidence)
        # tracker providing optical flow tracks
        self.tracker = LucasKanadeTracker(4 * self.detect_interval)
        # tracker people in video
        self.people = []
        # people that already passed in video
        self.count_passed = 0
        # current processed frame in video
        self.frame_idx = 0
        # total time in seconds the processing took
        self.total_time = 0.01
        if args.video:
            # use video file
            self.cap = cv.VideoCapture(args.video)
        else:
            # use webcam
            self.cap = cv.VideoCapture(0)

    def run(self):
        while True:
            # read frame from video source
            ret, frame = self.cap.read()
            if not ret:
                break

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

            # measure time
            stop = time.clock()
            self.total_time += stop - start

            # visualisation
            self.tracker.visualise(frame)
            self.visualise(frame)
            # wait for visualisation
            c = cv.waitKey(1)
            if c == 27: # esc press
                break
            elif c == 115: # s press
                # save visualialised image on s press
                cv.imwrite('vis-%s.png' % self.frame_idx, frame)

            self.frame_idx += 1

        # count also people currently in the frame
        self.frame_idx += self.finish_tracking_after * 2
        self._count_finished_tracks()
        # cleanup windows
        self.cap.release()
        cv.destroyAllWindows()
        print('passed people: ', self.count_passed)

    def count(self):
        """
        Updates tracks of people in video, counts finished tracks
        """
        self._add_new_people()
        self._count_finished_tracks()

    def _add_new_people(self):
        """
        Add new detections to existing tracks of people of create new ones
        """
        # find if we have new person in the image
        for new_person in self.detector.people:
            # try to match detected persons to the new ones
            matches = np.zeros(len(self.people))
            for i,person_track in enumerate(self.people):
                # count tracks that match
                matches[i] = matchRoisFromFlow(person_track[-1].roi, new_person,
                    self.tracker.tracks, self.frame_idx - person_track[-1].stamp)
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
            else:
                # we haven't found a matching person, lets add new one
                self.people.append([RectStamped(new_person, self.frame_idx)])

    def _count_finished_tracks(self):
        """
        Counts finished tracks (not updated for quite some time)
        """
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
        """
        Visualise tracks and optical flow
        """
        # print statistics in top left corner
        draw_str(frame, (20, 40), 'passed people: %d' % self.count_passed)
        draw_str(frame, (20, 60), 'fps: %d' % (self.frame_idx / self.total_time))
        if self.detector.other_objects:
            draw_str(frame, (20, 80), 'other objects: %s' % ', '.join(self.detector.other_objects))
        # draw people's tracks
        for track in self.people:
            # choose color from color palette
            color = randColor(hash(track[0]))
            # draw rectangle for each detection in track
            for roi, _ in track:
                cv.rectangle(frame, roi.tl(), roi.br(), color)

        # show visualisation in window
        cv.imshow("counter", frame)

# run main when this runs as a script
if __name__ == "__main__":
    main()
