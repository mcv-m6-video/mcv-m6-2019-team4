#!/usr/env/python3
"""
Copyright (c) 2019 LI RUOTENG
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import numpy as np
import matplotlib.pyplot as plt
from utils import flow_utils


"""
Modified on March 2019 in the framework of a MsC project. We only take the parts of the original script 'flowlib.py' concerning optical flow: read/write + visualization in both, KITTI format ('.png') and Middlebury ('.flo'), but not in '.pfm'. 
For the complete code, please refer to: https://github.com/liruoteng/OpticalFlowToolkit
All rights go to LI RUOTENG.
"""

import sys
sys.path.append('..')  # less hacky way unknown :$

# Define maximum/minimum flow values to avoid NaN/Inf
UNKNOWN_FLOW_THRESH = 1e7
SMALLFLOW = 0.0
TAU_MOTION = 3  # error smaller or equal than 3 pixels are not taken into account

"""
    Evaluation metrics for Optical Flow
"""


def squared_difference_noc(tu, tv, u, v, mask, plot=0, method='MSEN'):
    SEN = []
    # Compute squared difference
    E_du = tu - u
    E_dv = tv - v
    E = np.sqrt(E_du ** 2 + E_dv ** 2)

    # Set occluded pixels error to 0
    E[mask == 0] = 0

    # Plot histogram
    if (plot and method == 'MSEN') or (plot >= 1 and method == 'MSEN'):
        print('MSEN')
        plt.hist(E[mask == 1], bins=50, density=True)
        plt.xlabel('Difference (error)')
        plt.ylabel('Number of pixels (%)')
        plt.title('Squared difference for non-occluded pixels')
        plt.show()

        # TODO: create visualization for error (just plot E as an img? Or scale it somehow before)
        # TODO: estimated flow is more sparse than already sparse gt, improve visualization (more saturated colours,etc)
        # Check if we want to plot anything else
        if plot > 1:
            print('Preparing 3x1 plot of estimated flow-ground truth-error')
            # Need to use (h, w, channels) for flowlib.py functions
            gt_flow = np.array([tu, tv, mask]).transpose(2,1,0)
            aux_valid = np.ones(mask.shape)
            est_flow = np.array([u, v, aux_valid]).transpose(2,1,0)
            # E_valid = np.array([E, aux_valid])

            gt_flow_img = flow_utils.flow_to_image(gt_flow)
            est_flow_img = flow_utils.flow_to_image(est_flow)
            error_img = E.transpose(1,0)

            # fig, ax = plt.subplots(2, 1, sharex='col', sharey='row')
            # ax[0, 0] = plt.imshow(gt_flow_img)
            # ax[1, 0] = plt.imshow(est_flow_img)
            # ax[2, 0] = plt.
            plt.show()
            # TODO: test the above

    SEN = np.append(SEN, E[mask != 0])  # only non-occluded pixels

    return SEN

# TODO: MUST do the conversion inside the function and not at the evaluation function (astype(np.uint64))


def flow_error(tu, tv, u, v, mask, gt_value, method='EPE', tau=TAU_MOTION, plot=0):
    """
    * Modified on March 2019 to add MSEN(Mean Squared Error in Non-occluded areas) and	PEPN (Percentage of Erroneous
    Pixels in Non-occluded areas)
    - Added the possibility of inputting a mask of valid pixels that will be used in the computation (this is based off
     the original Matlab code from Stefan Roth ('flowAngErr_mask.m'), which additionally computed the mean angular error.
    - See 'eval_sequence(...)' below for more details.

    :param plot:        enable disable one (or more) plots. The options are as follows: '0' no plot; '1', plot histogram
    for MSEN; '>2' plot histogram (if MSEN) and figure with visual representation of the estimated, ground truth flow and
    error (3x1)
    :param tu: 			ground-truth horizontal flow map
    :param tv: 			ground-truth vertical flow map
    :param u:  			estimated horizontal flow map
    :param v:  			estimated vertical flow map
    :param mask:		validity mask
    :param gt_value:	value to reject pixels given 'mask' (0 or 1)
    :param method: 		error metric selected (def.:'EPE')
    :param tau:         minimum error threshold
    :return:			error measure computed

    """

    if method == 'EPE':  # notice that this is EPEall (for all pixels, occluded + non-occluded)
        """
        Calculate average end point error (a.k.a. Mean End Point Error: MEPE)
        :return: End point error of the estimated flow
        """
        smallflow = 0.0
        '''  We assume a border 'bord' equal to 0
        stu = tu[bord+1:end-bord,bord+1:end-bord]
        stv = tv[bord+1:end-bord,bord+1:end-bord]
        su = u[bord+1:end-bord,bord+1:end-bord]
        sv = v[bord+1:end-bord,bord+1:end-bord]
        '''
        stu = tu[:]
        stv = tv[:]
        su = u[:]
        sv = v[:]

        idxUnknow = (abs(stu) > UNKNOWN_FLOW_THRESH) | (abs(stv) > UNKNOWN_FLOW_THRESH) | mask[:] == gt_value
        stu[idxUnknow] = 0
        stv[idxUnknow] = 0
        su[idxUnknow] = 0
        sv[idxUnknow] = 0

        ind2 = [(np.absolute(stu) > SMALLFLOW) | (np.absolute(stv) > SMALLFLOW)]

        # Only used if we uncomment the """angle...""" to compute 'mean angular error (mang)'
        index_su = su[tuple(ind2)]
        index_sv = sv[tuple(ind2)]
        an = 1.0 / np.sqrt(index_su ** 2 + index_sv ** 2 + 1)
        un = index_su * an
        vn = index_sv * an

        index_stu = stu[tuple(ind2)]
        index_stv = stv[tuple(ind2)]
        tn = 1.0 / np.sqrt(index_stu ** 2 + index_stv ** 2 + 1)
        tun = index_stu * tn
        tvn = index_stv * tn

        '''
        angle = un * tun + vn * tvn + (an * tn)
        index = [angle == 1.0]
        angle[index] = 0.999
        ang = np.arccos(angle)
        mang = np.mean(ang)
        mang = mang * 180 / np.pi
        '''

        epe = np.sqrt((stu - su) ** 2 + (stv - sv) ** 2)
        epe = epe[tuple(ind2)]
        mepe = np.mean(epe)
        return mepe

    elif method == 'MSEN':
        # Compute squared difference for non-occluded pixels (mask == 1)
        SEN = squared_difference_noc(tu, tv, u, v, mask, plot=plot, method='MSEN')

        # Compute mean
        MSEN = np.mean(SEN)

        return MSEN

    elif method == 'PEPN':
        # Compute squared difference for non-occluded pixels (mask == 1)
        SEN = squared_difference_noc(tu, tv, u, v, mask, plot=plot, method='PEPN')

        # Compute percentage of erroneous pixels
        PEPN = (np.sum(SEN > tau) / len(SEN)) * 100

        return PEPN

    else:
        print("Non-valid error measure. Please, select one of the following: 'EPE', 'MSEN', 'PEPN'")
        return None


"""
    Compute metrics for a given (KITTI-formatted) sequence
"""


def eval_sequence(noc_path, est_path, val_path=''):

    flow_est = flow_utils.read_flow(est_path)
    u = flow_est[:, :, 0]
    v = flow_est[:, :, 1]
    # flow_est[:,:,2] is a vector of ones by default (ALL 'valid')

    # Load GT flows
    # Read flows in KITTI format
    flow_gt_noc = flow_utils.read_flow(noc_path)
    gt_u_noc = flow_gt_noc[:, :, 0]
    gt_v_noc = flow_gt_noc[:, :, 1]
    noc_mask = flow_gt_noc[:, :, 2].astype(np.uint64)

# Compute metrics for non-occluded pixels (EPEmat, MSEN, PEPN)
    gt_value = 0  # reject 0's in validity masks
    # EPEmat
    EPE_mat = flow_error(gt_u_noc, gt_v_noc, u, v, noc_mask, gt_value, method='EPE')
    # MSEN
    MSEN = flow_error(gt_u_noc, gt_v_noc, u, v, noc_mask, gt_value, method='MSEN', plot=2)
    # PEPN
    PEPN = flow_error(gt_u_noc, gt_v_noc, u, v, noc_mask, gt_value, method='PEPN', tau=3)

    # If the valid mask is provided, compute EPEumat, EPEall
    if len(val_path) > 0:
        flow_gt_val = flow_utils.read_flow(val_path)
        gt_u_val = flow_gt_val[:, :, 0]
        gt_v_val = flow_gt_val[:, :, 1]
        val_mask = flow_gt_val[:, :, 2].astype(np.uint64)

        # EPEall
        EPE_all = flow_error(gt_u_val, gt_v_val, u, v, val_mask, gt_value, method='EPE')  # don't count mask == 0
        # EPEumat
        # A "little" bit trickier. The occluded pixels have value 0 in noc_mask and 1 in occ_mask
        # Use logical operators with masks
        occ_mask = ~noc_mask & val_mask  # i.e.: occluded = not in non_occluded but are valid
        EPE_umat = flow_error(gt_u_val, gt_v_val, u, v, occ_mask, gt_value, method='EPE')

    # Print metrics
    print("Computed metrics:")
    print("\tEPEmat = {:.4f}\t MSEN = {:.4f}\t PEPN = {:.2f}%".format(EPE_mat, MSEN, PEPN))

    # Add additional metrics
    if len(val_path) > 0:
            print("\tEPEall = {:.4f}\t EPEumat = {:.4f}".format(EPE_all, EPE_umat))


if __name__ == '__main__':  # Testing a sequence
    # Path to data
    flow_noc_path = 'data/seq157/gt/noc/000157_10.png'  # only non-occluded pixels
    flow_val_path = 'data/seq157/gt/occ/000157_10.png'  # ALL valid (non-occ+occluded) pixels
    # Load the optical flow estimated via Lucas-Kanade
    flow_est_path = 'data/seq157/LKflow_000157_10.png'

    print("Testing ALL metrics for seq. 157 KITTI 2012...\n")
    eval_sequence(flow_noc_path, flow_est_path, flow_val_path)
    print("Testing finished successfully")
    print("\n")

    # Path to data (not in repo!)
    flow_noc_path = '../../devkit_kitti/matlab/data/flow_gt.png'
    flow_est_path = '../../devkit_kitti/matlab/data/flow_est.png'

    print("Testing ALL metrics for unknown KITTI testing sequence...\n")
    eval_sequence(flow_noc_path, flow_est_path)
    print("Testing finished successfully")
