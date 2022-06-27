import cv2
import numpy as np


def zoom_topleft(image):
    image_size = image.shape[:2]
    h, w = image_size

    quarter_image = image[:h//2, :w//2]

    scaled_image = cv2.resize(quarter_image, (w, h), interpolation=cv2.INTER_LINEAR)
    return scaled_image

def zoom_topright(image):
    image_size = image.shape[:2]
    h, w = image_size

    quarter_image = image[:h//2, w//2:]

    scaled_image = cv2.resize(quarter_image, (w, h), interpolation=cv2.INTER_LINEAR)
    return scaled_image

def zoom_botleft(image):
    image_size = image.shape[:2]
    h, w = image_size

    quarter_image = image[h//2:, :w//2]

    scaled_image = cv2.resize(quarter_image, (w, h), interpolation=cv2.INTER_LINEAR)
    return scaled_image

def zoom_botright(image):
    image_size = image.shape[:2]
    h, w = image_size

    quarter_image = image[h//2:, w//2:]

    scaled_image = cv2.resize(quarter_image, (w, h), interpolation=cv2.INTER_LINEAR)
    return scaled_image

def flip_horizontal(image):
    return cv2.flip(image, 0)

def flip_vertical(image):
    return cv2.flip(image, 1)


def zoom(image):
    image_size = image.shape
    scaled_up_image = cv2.resize(
        image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR
    )
    half_image = int(image_size[1] / 2)
    three_quarter = int(image_size[1] / 2) + image_size[1]
    return scaled_up_image[half_image:three_quarter, half_image:three_quarter]


def blur(image):
    return cv2.GaussianBlur(image, (7, 7), 0)


def invert(image):
    return 255 - image.copy()


def erode(image):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


def dilate(image, k=9):
    kernel = np.ones((k, k), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


def rotate90(img):
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)


def rotate180(img):
    return cv2.rotate(img, cv2.ROTATE_180)


def none(image):
    return image


distortions = [
    "flip_horizontal",
    "flip_vertical",
    "zoom",
    "blur",
    "invert",
    "erode",
    "dilate",
    "rotate90",
    "rotate180",
]

order_dependent_distortions = [
    "zoom",
    "blur",
    "erode",
    "dilate",
]

augment_distortions = [
    "flip_horizontal",
    "flip_vertical",
    "rotate90",
    "rotate180",
    "zoom",
    "blur",
    "erode",
    "dilate",
]

no_rep_distortions = [
    "flip_horizontal",
    "flip_vertical",
    "rotate90",
    "rotate180",

    "zoom_topleft",
    "zoom_topright",
    "zoom_botleft",
    "zoom_botright",

    "blur",
    "erode",
    "dilate",
]


rep_distortions = [
    "zoom_topleft",
    "zoom_topright",
    "zoom_botleft",
    "zoom_botright",
    "blur",
    "erode",
    "dilate",
]