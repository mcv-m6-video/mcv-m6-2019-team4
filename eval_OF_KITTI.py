#!/usr/env/python3

"""
Compute MSEN and PEPN for sequences 45 and 157 of the KITTI 2012 flow set.
"""
from utils import flow_utils
from evaluation import flow_metrics
import numpy as np
import matplotlib.pyplot as plt

def plot_optical_flow(image_file, flow):
    (h, w) = flow.shape[0:2]
    du = flow[:, :, 0]
    dv = flow[:, :, 1]
    valid = flow[:, :, 2]
    U = du * valid
    V = dv * valid
    #M = np.hypot(U, V)
    M = np.arctan2(U, V)
    
    X, Y = np.meshgrid(np.arange(0, w), np.arange(0, h))
    
    step = 10
    plt.figure()
    #plt.title("pivot='mid'; every third arrow; units='inches'")
    
    im = plt.imread(image_file)
    plt.imshow(im, cmap='gray')
   
    plt.quiver(X[::step, ::step], Y[::step, ::step], U[::step, ::step], V[::step, ::step], M[::step, ::step], 
               pivot='tail', units='xy', color='r', angles='xy', scale_units='xy', scale=.7)
    #plt.scatter(X[::step, ::step], Y[::step, ::step], color='r', s=2)
    plt.show()
    
    
if __name__ == '__main__':
    # Evaluate sequence 45
    # Load GT
    flow_noc_path = 'evaluation/data/seq45/gt/noc/000045_10.png'
    flow_gt_noc = flow_utils.read_flow(flow_noc_path)
    gt_u_noc = flow_gt_noc[:, :, 0]
    gt_v_noc = flow_gt_noc[:, :, 1]
    noc_mask = flow_gt_noc[:, :, 2]

    
    
    # Estimated
    flow_est_path = 'evaluation/data/seq45/LKflow_000045_10.png'
    flow_est = flow_utils.read_flow(flow_est_path)
    u = flow_est[:, :, 0]
    v = flow_est[:, :, 1]
    # flow_est[:,:,2] is a vector of ones by default (ALL 'valid')

    # Metrics
    # MSEN
    MSEN = flow_metrics.flow_error(gt_u_noc, gt_v_noc, u, v, noc_mask, 0, 'MSEN')

    # PEPN
    PEPN = flow_metrics.flow_error(gt_u_noc, gt_v_noc, u, v, noc_mask, 0, 'PEPN')

    # Print metrics
    print("Sequence 45 metrics:")
    print("MSEN = {:.4f}\t PEPN = {:.2f}%".format(MSEN, PEPN))

    visualize_flow(flow_gt_noc, 'RGB')
    plot_optical_flow('evaluation/data/seq45/000045_10.png', flow_gt_noc)

    # Evaluate sequence 157
    # Load GT
    flow_noc_path = 'evaluation/data/seq157/gt/noc/000157_10.png'
    flow_gt_noc = flow_utils.read_flow(flow_noc_path)
    gt_u_noc = flow_gt_noc[:, :, 0]
    gt_v_noc = flow_gt_noc[:, :, 1]
    noc_mask = flow_gt_noc[:, :, 2]

    # Estimated
    flow_est_path = 'evaluation/data/seq157/LKflow_000157_10.png'
    flow_est = flow_utils.read_flow(flow_est_path)
    u = flow_est[:, :, 0]
    v = flow_est[:, :, 1]
    # flow_est[:,:,2] is a vector of ones by default (ALL 'valid')

    # Metrics
    # MSEN
    MSEN = flow_metrics.flow_error(gt_u_noc, gt_v_noc, u, v, noc_mask, 0, 'MSEN')

    # PEPN
    PEPN = flow_metrics.flow_error(gt_u_noc, gt_v_noc, u, v, noc_mask, 0, 'PEPN')

# Print metrics
    print("Sequence 157 metrics:")
    print("MSEN = {:.4f}\t PEPN = {:.2f}%".format(MSEN, PEPN))

    visualize_flow(flow_gt_noc, 'RGB')
    plot_optical_flow('evaluation/data/seq157/000157_10.png', flow_gt_noc)