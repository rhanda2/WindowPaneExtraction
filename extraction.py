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
        self.MAX_QUAD_ANGLE_RANGE = MAX_ANGLE_RANGE

