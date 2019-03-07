#!/usr/env/python3

"""
Compute MSEN and PEPN for sequences 45 and 157 of the KITTI 2012 flow set.
"""
from utils import flow_utils, flow_visualization
from evaluation import flow_metrics
from paths import DATA_DIR

if __name__ == '__main__':
    # Evaluate sequence 45
    # Load GT
    flow_noc_path = DATA_DIR.joinpath('seq45/gt/noc/000045_10.png')
    flow_gt_noc = flow_utils.read_flow(flow_noc_path)
    gt_u_noc = flow_gt_noc[:, :, 0]
    gt_v_noc = flow_gt_noc[:, :, 1]
    noc_mask = flow_gt_noc[:, :, 2]

    # Estimated
    flow_est_path = DATA_DIR.joinpath('seq45/LKflow_000045_10.png')
    flow_est = flow_utils.read_flow(flow_est_path)
    u = flow_est[:, :, 0]
    v = flow_est[:, :, 1]
    # flow_est[:,:,2] is a vector of ones by default (ALL 'valid')

    # Metrics
    # MSEN
    MSEN = flow_metrics.flow_error(gt_u_noc, gt_v_noc, u, v, noc_mask, 0,
                                   'MSEN')

    # PEPN
    PEPN = flow_metrics.flow_error(gt_u_noc, gt_v_noc, u, v, noc_mask, 0,
                                   'PEPN')

    # Print metrics
    print("Sequence 45 metrics:")
    print("MSEN = {:.4f}\t PEPN = {:.2f}%".format(MSEN, PEPN))

    flow_visualization.plot_optical_flow(
        DATA_DIR.joinpath('seq45/000045_10.png'),
        flow_noc_path)

    # Evaluate sequence 157
    # Load GT
    flow_noc_path = DATA_DIR.joinpath('seq157/gt/noc/000157_10.png')
    flow_gt_noc = flow_utils.read_flow(flow_noc_path)
    gt_u_noc = flow_gt_noc[:, :, 0]
    gt_v_noc = flow_gt_noc[:, :, 1]
    noc_mask = flow_gt_noc[:, :, 2]

    # Estimated
    flow_est_path = DATA_DIR.joinpath('seq157/LKflow_000157_10.png')
    flow_est = flow_utils.read_flow(flow_est_path)
    u = flow_est[:, :, 0]
    v = flow_est[:, :, 1]
    # flow_est[:,:,2] is a vector of ones by default (ALL 'valid')

    # Metrics
    # MSEN
    MSEN = flow_metrics.flow_error(gt_u_noc, gt_v_noc, u, v, noc_mask, 0,
                                   'MSEN')

    # PEPN
    PEPN = flow_metrics.flow_error(gt_u_noc, gt_v_noc, u, v, noc_mask, 0,
                                   'PEPN')

    # Print metrics
    print("Sequence 157 metrics:")
    print("MSEN = {:.4f}\t PEPN = {:.2f}%".format(MSEN, PEPN))

    flow_visualization.plot_optical_flow(
        DATA_DIR.joinpath('seq157/000157_10.png'),
        flow_noc_path)
