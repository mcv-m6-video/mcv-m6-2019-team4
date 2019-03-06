from pathlib import Path

AICITY_DIR = Path('/home/jon/Datasets/MCV/video/AICity_data/train/S03/c010')
# AICITY_ANNOTATIONS = AICITY_DIR.joinpath('Anotation_40secs_AICITY_S03_C010.xml')
AICITY_ANNOTATIONS = Path(__file__).parent.joinpath('annotations.xml')
AICITY_ANNOTATIONS = Path(__file__).parent.joinpath('some_annotations.xml')
