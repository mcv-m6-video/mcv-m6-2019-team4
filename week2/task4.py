import cv2
from utils.background_substraction.background_substractor import analyze_sequence

def compare_color_spaces():


    methods =[
        'MOG2',
        'LSBP',
        'GMG',
        'GSOC',
        'CNT',
        'MOG',
        'Team4-Gaussian',
        'Team4-Adaptative'
    ]


    color_conversions =[
        cv2.COLOR_BGR2HSV,
        cv2.COLOR_BGR2Luv,
        cv2.COLOR_BGR2Lab,
        None
    ]



    for method in methods:
        for color_conversion in color_conversions:
            print ('Analyzing sequence with method: ' + method + ", and color conversion: " + str(color_conversion))
            analyze_sequence(method,color_conversion)