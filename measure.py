import argparse
import math
import numpy as np
import cv2
import os
import contours as poly_interactive
import extraction
from checkerboard import detect_checkerboard

def get_reference_corners(image):
    corners_list = []
    # resized = imutils.resize(image, width=600)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (5, 3), None)
    print(ret)
    # If found
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        for x in range(len(corners2)):
            corners_list.append(corners2[x][0])
    return corners_list


def get_relative_length(edge_pixels, refLength):
    ref = (refLength[1] + refLength[0]) / 2
    average_x = (edge_pixels[0] + edge_pixels[3]) / 2
    average_y = (edge_pixels[1] + edge_pixels[2]) / 2
    x = (average_x / ref)
    y = (average_y / ref)
    length = [x * 17 / 32, y * 17 / 32]
    return length


def get_dist_two_points(point1, point2):
    distX = abs(point1[0] - point2[0])
    distY = abs(point1[1] - point2[1])
    dist = math.sqrt(distX ** 2 + distY ** 2)
    return dist


def get_reference_length(image_path):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.imread(image_path)
    # print(image == None)
    # while(1):
    # cv2.imshow('image', image)
    print(image)
    ret, corners = cv2.findChessboardCorners(image, (9, 7), None)
    print('ret' + str(ret))
    # if ret:
    avg_horz = 0
    avg_virt = 0
    edges_horz = 0
    edges_virt = 0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners2 = cv2.cornerSubPix(image, corners, (11, 11), (-1, -1), criteria)
    for x in range(63):
        if ((x + 1) % 5) != 0:
            avg_virt += get_dist_two_points(corners2[x][0], corners2[x + 1][0])
            edges_virt += 1
        # if the point is in the row skip the comparison as there is no points on the left
        if (x + 5) <= 14:
            avg_horz += get_dist_two_points(corners2[x][0], corners2[x + 5][0])
            edges_horz += 1
    return (avg_horz / edges_horz), (avg_virt / edges_virt)
    # return 0, 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    # group.add_argument("--images", help="Directory of images to be scanned")
    group.add_argument("--image", help="Path to single image to be scanned")
    # ap.add_argument("-i", action='store_true',
    #                 help="Flag for manually verifying and/or setting document corners")

    args = vars(ap.parse_args())
    # im_dir = args["images"]
    im_file_path = args["image"]
    # interactive_mode = args["i"]

    extractor = extraction.Extractor()

    valid_formats = [".jpg", ".jpeg", ".jp2", ".png", ".bmp", ".tiff", ".tif"]

    get_ext = lambda f: os.path.splitext(f)[1].lower()

    # Scan single image specified by command line argument --image <IMAGE_PATH>

    # if im_file_path:
    extracted_img = extractor.scan(im_file_path)
    basename = os.path.basename(im_file_path)
    extracted_img_path = 'output' + '/' + basename
    # image = cv2.imread()
    x_ref, y_ref = get_reference_length(im_file_path)

    # print(x_ref)
    # print(y_ref)
    # print(x_ref == y_ref)
    # height, width = extracted_img.shape
    # print('Height ' + str(height))
    # print('Width ' + str(width))
    # Scan all valid images in directory specified by command line argument --images <IMAGE_DIR>
    # else:
    #     im_files = [f for f in os.listdir(im_dir) if get_ext(f) in valid_formats]
    #     for im in im_files:
    #         extractor.scan(im_dir + '/' + im)
    

