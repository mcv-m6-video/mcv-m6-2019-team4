#!/usr/env/python3

"""
Compute MSEN and PEPN for sequences 45 and 157 of the KITTI 2012 flow set.
"""
import numpy as np
from utils.flow_utils import read_flow, read_flo_file, read_png_file
from evaluation.flow_metrics import flow_error, squared_error_noc

if __name__ == '__main__':
    # Evaluate sequence 45
    # Load GT
    flow_noc_path = 'evaluation/data/seq45/gt/noc/000045_10.png'
    flow_gt_noc = read_flow(flow_noc_path)
    gt_u_noc = flow_gt_noc[:, :, 0]
    gt_v_noc = flow_gt_noc[:, :, 1]
    noc_mask = flow_gt_noc[:, :, 2].astype(np.uint64)  # non-occluded pixels have value 1

    # Estimated
    flow_est_path = 'evaluation/data/seq45/LKflow_000045_10.png'
    flow_est = read_flow(flow_est_path)
    u = flow_est[:, :, 0]
    v = flow_est[:, :, 1]
    # flow_est[:,:,2] is a vector of ones by default (ALL 'valid')

    # Metrics
    # MSEN
    MSEN = flow_error(gt_u_noc, gt_v_noc, u, v, noc_mask, 0, method='MSEN')

    # PEPN
    PEPN = flow_error(gt_u_noc, gt_v_noc, u, v, noc_mask, 0, method='PEPN')

    # Print metrics
    print("Sequence 45 metrics:")
    print("MSEN = {:.4f}\t PEPN = {:.2f}%".format(MSEN, PEPN))

    # Evaluate sequence 157
    # Load GT
    flow_noc_path = 'evaluation/data/seq157/gt/noc/000157_10.png'
    flow_gt_noc = read_flow(flow_noc_path)
    gt_u_noc = flow_gt_noc[:, :, 0]
    gt_v_noc = flow_gt_noc[:, :, 1]
    noc_mask = flow_gt_noc[:, :, 2].astype(np.uint64)  # non-occluded pixels have value 1

    # Estimated
    flow_est_path = 'evaluation/data/seq157/LKflow_000157_10.png'
    flow_est = read_flow(flow_est_path)
    u = flow_est[:, :, 0]
    v = flow_est[:, :, 1]
    # flow_est[:,:,2] is a vector of ones by default (ALL 'valid')

    # Metrics
    # MSEN
    MSEN = flow_error(gt_u_noc, gt_v_noc, u, v, noc_mask, 0, method='MSEN')

    # PEPN
    PEPN = flow_error(gt_u_noc, gt_v_noc, u, v, noc_mask, 0, method='PEPN')

# Print metrics
    print("Sequence 45 metrics:")
    print("MSEN = {:.4f}\t PEPN = {:.2f}%".format(MSEN, PEPN))

