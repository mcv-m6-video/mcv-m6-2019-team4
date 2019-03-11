
import cv2
from utils.background_substraction.background_substractor import analyze_sequence

def compare_state_of_the_art():


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


    for method in methods:
        print ('Analyzing sequence with method: ' + method)
        analyze_sequence(method, None)

