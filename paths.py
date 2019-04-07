from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
AICITY_DIR = PROJECT_ROOT.joinpath('data', 'AICity_data', 'train', 'S03',
                                   'c010')
AICITY_ANNOTATIONS = PROJECT_ROOT.joinpath('full_annotations.xml')

DATA_DIR = PROJECT_ROOT.joinpath('data')

AICITY_MC_ROOT = PROJECT_ROOT.joinpath('..', 'data', 'aic19-track1-mtmc-train')
