from background_substractor import analyze_sequence


def compare_state_of_the_art():
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

    for method in methods:
        print(f'Analyzing sequence with method: {method}')
        analyze_sequence(method, None)
