from pyimagesearch import transform
from pyimagesearch import imutils
from scipy.spatial import distance as dist
from matplotlib.patches import Polygon
import contours as poly_interactive
import numpy as np
import matplotlib.pyplot as plt
import itertools
import math
import cv2
from pylsd.lsd import lsd

import argparse
import os

class Extractor(object):
    """This object extracts the window frame from the image"""

    def __init__(self, interactive=True, MIN_WINDOW_AREA_RATIO=0.25, MAX_ANGLE_RANGE=40):
        """
        Args:
            interactive (boolean): If True, user can adjust screen contour before
                transformation occurs in interactive pyplot window.
            MIN_WINDOW_AREA_RATIO (float): A contour will be rejected if its corners
                do not form a quadrilateral that covers at least MIN_WINDOW_AREA_RATIO
                of the original image. Defaults to 0.25. Ensures that the window is
                big enough in the photo
            MAX_ANGLE_RANGE (int):  A contour will also be rejected if the range
                        of its interior angles exceeds MAX_ANGLE_RANGE. Defaults to 40.
                        Checks if the photos is not too distorted.
        """
        self.interactive = interactive
        self.MIN_WINDOW_AREA_RATIO = MIN_WINDOW_AREA_RATIO
        self.MAX_ANGLE_RANGE = MAX_ANGLE_RANGE

    def filter_corners(self, corners, min_dist=20):
        """Getting corners more than a certain distance to each other"""
        def is_far_enough(listc, corner):
            return all(dist.euclidean(item, corner) >= min_dist for item in listc)

        filtered_corners = []
        for c in corners:
            if is_far_enough(filtered_corners, c):
                filtered_corners.append(c)
        return filtered_corners

    def get_angle(self, point1, point2, point3):
        """Returns angle between the two lines with point2 as vertex in degrees
         using vector dot product"""
        a = np.radians(np.array(point1))
        b = np.radians(np.array(point2))
        c = np.radians(np.array(point3))

        ab_ray = a - b
        cb_ray = c - b

        return np.degrees(
            math.acos(np.dot(ab_ray, cb_ray) / (np.linalg.norm(ab_ray) * np.linalg.norm(cb_ray))))

    def get_angle_range(self, quad):
        """
        :param quad: The quadrilateral ABCD
        :return: Returns the range between the largest and smallest interior angles of the quadrilateral
        """

        a, b, c, d = quad
        angle_a = self.get_angle(d[0], a[0], b[0])
        angle_b = self.get_angle(a[0], b[0], c[0])
        angle_c = self.get_angle(b[0], c[0], d[0])
        angle_d = self.get_angle(c[0], d[0], a[0])

        angles = [angle_a, angle_b, angle_c, angle_d]
        return np.ptp(angles)

    def get_corners(self, img):
        """
        It should return at most 10 corners if proper preprocessing is met.
        :param img: The input image is expected to be rescaled and Canny filtered
        :return: Returns a list of corners ((x,y) tuples) found in the input image.
        """
        lines = lsd(img)

        corners = []
        if lines is not None:
            # separate out the horizontal and vertical lines, and draw them back onto separate canvases
            lines = lines.squeeze().astype(np.int32).tolist()
            horizontal_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            vertical_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            for line in lines:
                x1, y1, x2, y2, _ = line
                # uses relative change in vertical or horizontal directions to find the type of ine
                if abs(x2 - x1) > abs(y2 - y1):
                    (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda pt: pt[0])
                    cv2.line(horizontal_lines_canvas, (max(x1 - 5, 0), y1), (min(x2 + 5, img.shape[1] - 1), y2), 255, 2)
                else:
                    (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda pt: pt[1])
                    cv2.line(vertical_lines_canvas, (x1, max(y1 - 5, 0)), (x2, min(y2 + 5, img.shape[0] - 1)), 255, 2)

            lines = []

            # find the horizontal lines (connected-components -> bounding boxes -> final lines)
            (contours, hierarchy) = cv2.findContours(horizontal_lines_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:2]
            horizontal_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            for contour in contours:
                contour = contour.reshape((contour.shape[0], contour.shape[2]))
                min_x = np.amin(contour[:, 0], axis=0) + 2
                max_x = np.amax(contour[:, 0], axis=0) - 2
                left_y = int(np.average(contour[contour[:, 0] == min_x][:, 1]))
                right_y = int(np.average(contour[contour[:, 0] == max_x][:, 1]))
                lines.append((min_x, left_y, max_x, right_y))
                cv2.line(horizontal_lines_canvas, (min_x, left_y), (max_x, right_y), 1, 1)
                corners.append((min_x, left_y))
                corners.append((max_x, right_y))

            # find the vertical lines (connected-components -> bounding boxes -> final lines)
            (contours, hierarchy) = cv2.findContours(vertical_lines_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:2]
            vertical_lines_canvas = np.zeros(img.shape, dtype=np.uint8).tolist()
            for contour in contours:
                contour = contour.reshape((contour.shape[0], contour.shape[2]))
                min_y = np.amin(contour[:, 1], axis=0) + 2
                max_y = np.amax(contour[:, 1], axis=0) - 2
                top_x = int(np.average(contour[contour[:, 1] == min_y][:, 0]))
                bottom_x = int(np.average(contour[contour[:, 1] == max_y][:, 0]))
                lines.append((top_x, min_y, bottom_x, max_y))
                cv2.line(vertical_lines_canvas, (top_x, min_y), (bottom_x, max_y), 1, 1)
                corners.append((top_x, min_y))
                corners.append((bottom_x, max_y))

            # find the corners
            corners_y, corners_x = np.where(horizontal_lines_canvas + vertical_lines_canvas == 2)
            corners += zip(corners_x, corners_y)
