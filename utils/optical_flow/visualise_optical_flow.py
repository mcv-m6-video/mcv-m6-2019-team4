from optical_flow import flow_utils
from paths import DATA_DIR


def run():
    # Load example sequence
    # Path to data
    flow_noc_path = DATA_DIR.joinpath(
        'seq157/gt/noc/000157_10.png')  # only non-occluded pixels
    flow_val_path = DATA_DIR.joinpath(
        'seq157/gt/occ/000157_10.png')  # ALL valid (non-occ+occluded) pixels

    # Load the optical flow estimated via Lucas-Kanade
    flow_est_path = DATA_DIR.joinpath('seq157/LKflow_000157_10.png')

    # Estimated
    flow_utils.show_flow(flow_est_path)

    # Estimated + ground truth (side-by-side)
    flow_utils.show_flow_pair(flow_est_path, flow_noc_path)


if __name__ == '__main__':
    run()
