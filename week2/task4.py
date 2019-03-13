import cv2
from utils import background_substractor


def compare_color_spaces():
    methods = [
        'MOG2',
        'LSBP',
        'GMG',
        'GSOC',
        'CNT',
        'MOG',
        'Team4-Gaussian',
        'Team4-Adaptative'
    ]

    color_conversions = [
        cv2.COLOR_BGR2HSV,
        cv2.COLOR_BGR2Luv,
        cv2.COLOR_BGR2Lab,
        None
    ]

    for method in methods:
        for color_conversion in color_conversions:
            print(
                f"Analyzing sequence with method: {method}, "
                f"and color conversion: {color_conversion}")
            background_substractor.analyze_sequence(method, color_conversion)
