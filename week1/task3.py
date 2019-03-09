""" Task 3: Optical flow evaluation metrics. """
from utils import optical_flow
from evaluation import optical_flow as flow_eval
from paths import DATA_DIR

""" T3.1 MSEN & T3.2 PEPN """


def compute_msen_and_pepn_over_kitti():
    """
    Compute MSEN and PEPN for sequences 45 and 157 of the KITTI 2012 flow set.
    """
    # Evaluate sequence 45
    # Load GT
    flow_noc_path = DATA_DIR.joinpath('seq45/gt/noc/000045_10.png')
    flow_gt_noc = optical_flow.read_flow(flow_noc_path)
    gt_u_noc = flow_gt_noc[:, :, 0]
    gt_v_noc = flow_gt_noc[:, :, 1]
    noc_mask = flow_gt_noc[:, :, 2]

    # Estimated
    flow_est_path = DATA_DIR.joinpath('seq45/LKflow_000045_10.png')
    flow_est = optical_flow.read_flow(flow_est_path)
    u = flow_est[:, :, 0]
    v = flow_est[:, :, 1]
    # flow_est[:,:,2] is a vector of ones by default (ALL 'valid')

    # Metrics
    # MSEN
    MSEN = flow_eval.flow_error(gt_u_noc, gt_v_noc, u, v, noc_mask, 0,
                                'MSEN')

    # PEPN
    PEPN = flow_eval.flow_error(gt_u_noc, gt_v_noc, u, v, noc_mask, 0,
                                'PEPN')

    # Print metrics
    print("Sequence 45 metrics:")
    print("MSEN = {:.4f}\t PEPN = {:.2f}%".format(MSEN, PEPN))

    optical_flow.plot_optical_flow(
        DATA_DIR.joinpath('seq45/000045_10.png'),
        flow_noc_path)

    # Evaluate sequence 157
    # Load GT
    flow_noc_path = DATA_DIR.joinpath('seq157/gt/noc/000157_10.png')
    flow_gt_noc = optical_flow.read_flow(flow_noc_path)
    gt_u_noc = flow_gt_noc[:, :, 0]
    gt_v_noc = flow_gt_noc[:, :, 1]
    noc_mask = flow_gt_noc[:, :, 2]

    # Estimated
    flow_est_path = DATA_DIR.joinpath('seq157/LKflow_000157_10.png')
    flow_est = optical_flow.read_flow(flow_est_path)
    u = flow_est[:, :, 0]
    v = flow_est[:, :, 1]
    # flow_est[:,:,2] is a vector of ones by default (ALL 'valid')

    # Metrics
    # MSEN
    MSEN = flow_eval.flow_error(gt_u_noc, gt_v_noc, u, v, noc_mask, 0,
                                'MSEN')

    # PEPN
    PEPN = flow_eval.flow_error(gt_u_noc, gt_v_noc, u, v, noc_mask, 0,
                                'PEPN')

    # Print metrics
    print("Sequence 157 metrics:")
    print("MSEN = {:.4f}\t PEPN = {:.2f}%".format(MSEN, PEPN))

    optical_flow.plot_optical_flow(
        DATA_DIR.joinpath('seq157/000157_10.png'),
        flow_noc_path)


""" T3.1 MMEN & T3.2 PPEN """

""" T3.3 Analysis & Visualizations (baselines) """


def test_optical_flow_metrics():
    # Path to data
    flow_noc_path = DATA_DIR.joinpath(
        'seq157/gt/noc/000157_10.png')  # only non-occluded pixels
    flow_val_path = DATA_DIR.joinpath(
        'seq157/gt/occ/000157_10.png')  # ALL valid (non-occ+occluded) pixels
    # Load the optical flow estimated via Lucas-Kanade
    flow_est_path = DATA_DIR.joinpath('seq157/LKflow_000157_10.png')

    print("Testing ALL metrics for seq. 157 KITTI 2012...\n")
    flow_eval.eval_sequence(flow_noc_path,
                            flow_est_path,
                            flow_val_path)
    print("Testing finished successfully")
    print("\n")

    # Path to data (not in repo!)
    flow_noc_path = '../../devkit_kitti/matlab/data/flow_gt.png'
    flow_est_path = '../../devkit_kitti/matlab/data/flow_est.png'

    print("Testing ALL metrics for unknown KITTI testing sequence...\n")
    flow_eval.eval_sequence(flow_noc_path, flow_est_path)
    print("Testing finished successfully")
