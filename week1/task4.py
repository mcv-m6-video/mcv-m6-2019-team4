""" Task 4: Visual representation optical flow. """
from paths import DATA_DIR
from utils import optical_flow

""" T4 Optical flow plot """


def visualise_optical_flow():
    # Load example sequence
    # Path to data
    flow_noc_path = DATA_DIR.joinpath(
        'seq157/gt/noc/000157_10.png')  # only non-occluded pixels
    flow_val_path = DATA_DIR.joinpath(
        'seq157/gt/occ/000157_10.png')  # ALL valid (non-occ+occluded) pixels

    # Load the optical flow estimated via Lucas-Kanade
    flow_est_path = DATA_DIR.joinpath('seq157/LKflow_000157_10.png')

    # Estimated
    optical_flow.show_flow(flow_est_path)

    # Estimated + ground truth (side-by-side)
    optical_flow.show_flow_pair(flow_est_path, flow_noc_path)
