import os as _os
import os.path as _path
import numpy as np
import cv2 as _cv2
from PIL import ImageFont
import numpy as _np
from hashlib import md5 as _md5

_LOC = _path.realpath(_path.join(_os.getcwd(),_path.dirname(__file__)))

#https://clrs.cc/
_COLOR_NAME_TO_RGB = dict(
    navy=((0, 38, 63), (255, 255, 255)),
    blue=((0, 120, 210), (255, 255, 255)),
    aqua=((115, 221, 252), (255, 255, 255)),
    teal=((15, 205, 202), (255, 255, 255)),
    olive=((52, 153, 114), (255, 255, 255)),
    green=((0, 204, 84), (255, 255, 255)),
    lime=((1, 255, 127), (255, 255, 255)),
    yellow=((255, 216, 70), (255, 255, 255)),
    orange=((255, 125, 57), (255, 255, 255)),
    red=((255, 47, 65), (255, 255, 255)),
    maroon=((135, 13, 75), (255, 255, 255)),
    fuchsia=((246, 0, 184), (255, 255, 255)),
    purple=((179, 17, 193), (255, 255, 255)),
    black=((24, 24, 24), (255, 255, 255)),
    gray=((168, 168, 168), (255, 255, 255)),
    silver=((220, 220, 220), (255, 255, 255))
)

_COLOR_NAMES = list(_COLOR_NAME_TO_RGB)

_DEFAULT_COLOR_NAME = "green"

_FONT_PATH = _os.path.join(_LOC, "Ubuntu-B.ttf")

def _rgb_to_bgr(color):
    return list(reversed(color))

def _color_image(image, font_color, background_color):
    return background_color + (font_color - background_color) * image / 255

def _get_label_image(text, font_color_tuple_bgr, background_color_tuple_bgr, font):
    text_image = font.getmask(text)
    shape = list(reversed(text_image.size))
    bw_image = np.array(text_image).reshape(shape)

    image = [
        _color_image(bw_image, font_color, background_color)[None, ...]
        for font_color, background_color
        in zip(font_color_tuple_bgr, background_color_tuple_bgr)
    ]

    return np.concatenate(image).transpose(1, 2, 0)

def draw_bounding_box(image, top_left, bottom_right = None, width_height = None, label = None, color = None):
    '''
    https://github.com/nalepae/bounding-box + few changes
    '''
    assert bottom_right or width_height, f'\n\nAtleast one of `bottom_right` or `width_height` is required, got both None.\n'

    left = int(top_left[0])
    top = int(top_left[1])

    if bottom_right:
        right = int(bottom_right[0])
        bottom = int(bottom_right[1])
    else:
        right = int(left + width_height[0])
        bottom = int(top + width_height[1])

    _FONT_HEIGHT = int(max(image.shape[0], image.shape[1]) ** 0.44)
    _FONT = ImageFont.truetype(_FONT_PATH, _FONT_HEIGHT)

    if type(image) is not _np.ndarray:
        raise TypeError("'image' parameter must be a numpy.ndarray")
    try:
        left, top, right, bottom = int(left), int(top), int(right), int(bottom)
    except ValueError:
        raise TypeError("'left', 'top', 'right' & 'bottom' must be a number")

    if label and type(label) is not str:
        raise TypeError("'label' must be a str")

    if label and not color:
        hex_digest = _md5(label.encode()).hexdigest()
        color_index = int(hex_digest, 16) % len(_COLOR_NAME_TO_RGB)
        color = _COLOR_NAMES[color_index]

    if not color:
        color = _DEFAULT_COLOR_NAME

    if type(color) is not str:
        raise TypeError("'color' must be a str")

    if color not in _COLOR_NAME_TO_RGB:
        msg = "'color' must be one of " + ", ".join(_COLOR_NAME_TO_RGB)
        raise ValueError(msg)

    colors = [_rgb_to_bgr(item) for item in _COLOR_NAME_TO_RGB[color]]
    color, color_text = colors

    _cv2.rectangle(image, (left, top), (right, bottom), color, 2)

    border_width = int(max(right - left, bottom - top) ** 0.34)
    border_length = int(max(right - left, bottom - top) * 0.1)

    _cv2.line(image, (right - border_length, top), (right, top), color, border_width)
    _cv2.line(image, (right, top), (right, top + border_length), color, border_width)

    _cv2.line(image, (left, bottom - border_length), (left, bottom), color, border_width)
    _cv2.line(image, (left, bottom), (left + border_length, bottom), color, border_width)

    _cv2.line(image, (right, bottom - border_length), (right, bottom), color, border_width)
    _cv2.line(image, (right, bottom), (right - border_length, bottom), color, border_width)

    if label:
        _, image_width, _ = image.shape

        label_image =  _get_label_image(label, color_text, color, _FONT)
        label_height, label_width, _ = label_image.shape

        rectangle_height, rectangle_width = int(1.26 * label_height), int(1.05 * label_width)

        rectangle_bottom = top
        rectangle_left = max(0, min(left - 1, image_width - rectangle_width))

        rectangle_top = rectangle_bottom - rectangle_height
        rectangle_right = rectangle_left + rectangle_width

        label_top = rectangle_top + 1

        if rectangle_top < 0:
            rectangle_top = top
            rectangle_bottom = rectangle_top + label_height + 1

            label_top = rectangle_top

        label_left = rectangle_left + 1
        label_bottom = label_top + label_height
        label_right = label_left + label_width

        rec_left_top = (rectangle_left, int(rectangle_top * 0.98))
        rec_right_bottom = (rectangle_right, rectangle_bottom)

        _cv2.rectangle(image, rec_left_top, rec_right_bottom, color, -1)

        image[label_top:label_bottom, label_left:label_right, :] = label_image