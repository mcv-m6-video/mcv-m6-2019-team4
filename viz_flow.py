from utils import flow_utils


if __name__ == '__main__':
    # Load example sequence
    # Path to data
    flow_noc_path = 'data/seq157/gt/noc/000157_10.png'  # only non-occluded pixels
    flow_val_path = 'data/seq157/gt/occ/000157_10.png'  # ALL valid (non-occ+occluded) pixels

    # Load the optical flow estimated via Lucas-Kanade
    flow_est_path = 'data/seq157/LKflow_000157_10.png'

    # Estimated
    flow_utils.show_flow(flow_est_path)

    # Estimated + ground truth (side-by-side)
    flow_utils.show_flow_pair(flow_est_path, flow_noc_path)
