from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
AICITY_DIR = PROJECT_ROOT.joinpath('data', 'AICity_data', 'train', 'S03', 'c010')
# AICITY_ANNOTATIONS = AICITY_DIR.joinpath('Anotation_40secs_AICITY_S03_C010.xml')
AICITY_ANNOTATIONS = PROJECT_ROOT.joinpath('annotations.xml')
AICITY_ANNOTATIONS = PROJECT_ROOT.joinpath('some_annotations.xml')

DATA_DIR = PROJECT_ROOT.joinpath('data')
