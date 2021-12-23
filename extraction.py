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

    def __init__(self, MIN_WINDOW_AREA_RATIO=0.25, MAX_ANGLE_RANGE=25):
        """
        Args:            
            MIN_WINDOW_AREA_RATIO (float): A contour will be rejected if its corners
                do not form a quadrilateral that covers at least MIN_WINDOW_AREA_RATIO
                of the original image. Defaults to 0.25. Ensures that the window is
                big enough in the photo
            MAX_ANGLE_RANGE (int):  A contour will also be rejected if the range
                        of its interior angles exceeds MAX_ANGLE_RANGE. Defaults to 25.
                        Checks if the photos is not too distorted.
        """       
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
            thickness = 4
            for line in lines:
                x1, y1, x2, y2, _ = line
                # uses relative change in vertical or horizontal directions to find the type of ine
                if abs(x2 - x1) > abs(y2 - y1):
                    (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda pt: pt[0])
                    cv2.line(horizontal_lines_canvas, (max(x1 - 3, 0), y1), (min(x2 + 3, img.shape[1] - 1), y2), 255, thickness)
                else:
                    (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda pt: pt[1])
                    cv2.line(vertical_lines_canvas, (x1, max(y1 - 3, 0)), (x2, min(y2 + 3, img.shape[0] - 1)), 255, thickness)

            lines = []

            # find the horizontal lines (connected-components -> bounding boxes -> final lines)
            (contours, hierarchy) = cv2.findContours(horizontal_lines_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # sorting based on arc length
            contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:2]
            horizontal_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            for contour in contours:
                contour = contour.reshape((contour.shape[0], contour.shape[2]))
                min_x = np.amin(contour[:, 0], axis=0) + thickness
                max_x = np.amax(contour[:, 0], axis=0) - thickness
                left_y = int(np.average(contour[contour[:, 0] == min_x][:, 1]))
                right_y = int(np.average(contour[contour[:, 0] == max_x][:, 1]))
                lines.append((min_x, left_y, max_x, right_y))
                cv2.line(horizontal_lines_canvas, (min_x, left_y), (max_x, right_y), 1, 1)
                corners.append((min_x, left_y))
                corners.append((max_x, right_y))

            # find the vertical lines (connected-components -> bounding boxes -> final lines)
            (contours, hierarchy) = cv2.findContours(vertical_lines_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:2]
            vertical_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            for contour in contours:
                contour = contour.reshape((contour.shape[0], contour.shape[2]))
                min_y = np.amin(contour[:, 1], axis=0) + thickness
                max_y = np.amax(contour[:, 1], axis=0) - thickness
                top_x = int(np.average(contour[contour[:, 1] == min_y][:, 0]))
                bottom_x = int(np.average(contour[contour[:, 1] == max_y][:, 0]))
                lines.append((top_x, min_y, bottom_x, max_y))
                cv2.line(vertical_lines_canvas, (top_x, min_y), (bottom_x, max_y), 1, 1)
                corners.append((top_x, min_y))
                corners.append((bottom_x, max_y))

            # find the corners (in both horizontal and vertical lines)
            corners_y, corners_x = np.where(horizontal_lines_canvas + vertical_lines_canvas == 2)
            corners += zip(corners_x, corners_y)

            corners = self.filter_corners(corners)
            return corners

    def is_valid_contour(self, cnt, IM_WIDTH, IM_HEIGHT):
        """:return: Returns True if the contour is within min quad area and max angle requirements"""

        return (len(cnt) == 4 and cv2.contourArea(cnt) > IM_WIDTH * IM_HEIGHT * self.MIN_WINDOW_AREA_RATIO
                and self.get_angle_range(cnt) < self.MAX_ANGLE_RANGE)

    def get_contour(self, rescaled_image):
        """
        It considers the corners returned from get_corners() and uses heuristics
        to chooses four corners that most likely represent the corners of the
        document. If no accurate contour is found the corners of the image is returned.

        :return: Returns a numpy array of shape (4, 2) containing the vertices of the four corners of the document in the image
        """
        MORPH = 10
        CANNY = 80
        HOUGH = 25

        IM_HEIGHT, IM_WIDTH, _ = rescaled_image.shape

        # convert the image to grayscale and blur it slightly
        gray = cv2.cvtColor(rescaled_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # dilate helps to remove potential holes between edge segments
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH, MORPH))
        dilated = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

        # find edges and mark them in the output map using the Canny algorithm
        edged = cv2.Canny(dilated, 0, CANNY)
        test_corners = self.get_corners(edged)

        approx_contours = []

        if len(test_corners) >= 4:
            quads = []

            for quad in itertools.combinations(test_corners, 4):
                points = np.array(quad)
                points = transform.order_points(points)
                points = np.array([[p] for p in points], dtype="int32")
                quads.append(points)

            # get top five quadrilaterals by area
            quads = sorted(quads, key=cv2.contourArea, reverse=True)[:5]
            # sort candidate quadrilaterals by their angle range, which helps remove outliers
            quads = sorted(quads, key=self.get_angle_range)

            approx = quads[0]
            if self.is_valid_contour(approx, IM_WIDTH, IM_HEIGHT):
                approx_contours.append(approx)

            # for debugging: uncomment the code below to draw the corners and countour found
            # by get_corners() and overlay it on the image

            # cv2.drawContours(rescaled_image, [approx], -1, (20, 20, 255), 2)
            # plt.scatter(*zip(*test_corners))
            # plt.imshow(rescaled_image)
            # plt.show()

        # also attempt to find contours directly from the edged image, which occasionally
        # produces better results
        (cnts, hierarchy) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

        # loop over the contours
        for c in cnts:
            # approximate the contour
            approx = cv2.approxPolyDP(c, 80, True)
            if self.is_valid_contour(approx, IM_WIDTH, IM_HEIGHT):
                approx_contours.append(approx)
                break

        # If we did not find any valid contours, just use the whole image
        if not approx_contours:
            TOP_RIGHT = (IM_WIDTH, 0)
            BOTTOM_RIGHT = (IM_WIDTH, IM_HEIGHT)
            BOTTOM_LEFT = (0, IM_HEIGHT)
            TOP_LEFT = (0, 0)
            screenCnt = np.array([[TOP_RIGHT], [BOTTOM_RIGHT], [BOTTOM_LEFT], [TOP_LEFT]])

        else:
            screenCnt = max(approx_contours, key=cv2.contourArea)

        return screenCnt.reshape(4, 2)

    def interactive_get_contour(self, screenCnt, rescaled_image):
        poly = Polygon(screenCnt, animated=True, fill=False, color="yellow", linewidth=5)
        fig, ax = plt.subplots()
        ax.add_patch(poly)
        ax.set_title(('Drag the corners of the box to the corners of the document. \n'
                      'Close the window when finished.'))
        p = poly_interactive.PolygonInteractor(ax, poly)
        plt.imshow(rescaled_image)
        plt.show()

        new_points = p.get_poly_points()[:4]
        new_points = np.array([[p] for p in new_points], dtype="int32")
        return new_points.reshape(4, 2)

    def scan(self, image_path):

        RESCALED_HEIGHT = 500.0
        OUTPUT_DIR = 'output'

        # load the image and compute the ratio of the old height
        # to the new height, clone it, and resize it
        image = cv2.imread(image_path)

        assert (image is not None)

        ratio = image.shape[0] / RESCALED_HEIGHT
        orig = image.copy()
        rescaled_image = imutils.resize(image, height=int(RESCALED_HEIGHT))

        # get the contour of the document
        screenCnt = self.get_contour(rescaled_image)

        
        screenCnt = self.interactive_get_contour(screenCnt, rescaled_image)

        # apply the perspective transformation
        warped = transform.four_point_transform(orig, screenCnt * ratio)

        # convert the warped image to grayscale
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        # sharpen image
        sharpen = cv2.GaussianBlur(gray, (0, 0), 3)
        # sharpen = cv2.addWeighted(gray, 1.5, sharpen, -0.5, 0)

        # apply adaptive threshold to get black and white effect
        # thresh = cv2.adaptiveThreshold(sharpen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15)

        # save the transformed image
        basename = os.path.basename(image_path)
        cv2.imwrite(OUTPUT_DIR + '/' + basename, sharpen)
        print("Proccessed " + basename)

        return sharpen

