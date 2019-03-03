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

"""
Modified on March 2019 in the framework of a MsC project. We only take the parts of the original script 'flowlib.py' concerning optical flow: read/write + visualization in both, KITTI format ('.png') and Middlebury ('.flo'), but not in '.pfm'. 
For the complete code, please refer to: https://github.com/liruoteng/OpticalFlowToolkit
All rights go to LI RUOTENG.
"""

import numpy as np
import png
import cv2
from .utils.flow_metrics import read_flow, read_flo_file, read_png_file

# Define maximum/minimum flow values to avoid NaN/Inf
UNKNOWN_FLOW_THRESH = 1e7
SMALLFLOW = 0.0
TAU_MOTION = 3  # error smaller or equal than 3 pixels are not taken into account

"""
	Evaluation metrics for Optical Flow
"""

def flow_error(tu, tv, u, v, mask, gt_value, method='EPE'):
	"""
	Modified on March 2019 to add MSE(Mean Squared Error) in Non-occluded areas (MSEN) and
	PEPN (Percentage of Erroneous Pixels in Non-occluded areas)
	Also added the possibility of passing in a mask of valid pixels and or occluded ones
	So the input mask 'mask' may be tha combination of both, as follows:
	valid_mask: 1s valid, 0s invalid; occ_mask: 1s occluded, 0s non-occluded
	We want valid_mask=1 but we can choose how we provide the occlusion mask to compute only occluded
	pixels or only non-occluded ones. See 'test_error' below for more details.

	:param tu: 			ground-truth horizontal flow map
	:param tv: 			ground-truth vertical flow map
	:param u:  			estimated horizontal flow map
	:param v:  			estimated vertical flow map
	:param mask:		combination of validity and occlusion mask (target validity value=1(i.e.: valid); target occlusion mask value is 0 for MSEN and PEPN, 0 or 1 for EPE if we ONLY want to compute occluded or non-occluded values)
	:param gt_value:	value to reject pixels given 'mask' (0 or 1)
	:param method: 		error metric selected (def.:'EPE')
	NOTE: to compute EPE for all pixels, just input validity mask w/o occlusions
	:return:			error measure computed 
	
	"""

	# Compute 

	if method == 'EPE':	  # notice that this is EPEall (for all pixels, occluded + non-occluded)
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
		# Differences in horizontal and vertical
		E_du = tu - u
		E_dv = tv - v
		# Add together
		SE = E_du**2 + E_dv**2
		# Compute mean    
		MSE = np.mean(SE)

		# Exclude non-valid and occluded pixels
		MSE[mask==1] = 0  # valid AND non-occluded ==> 1 & 0 = 0 (so we want to avoid 1's)
		MSEN = MSE

		return MSEN

	elif method == 'PEPN':
		# Differences in horizontal and vertical
		E_du = tu - u
		E_dv = tv - v
		# Add together
		SE = E_du**2 + E_dv**2

		# Exclude non-valid and occluded pixels
		MSE[mask==1] = 0

		PEPN = len([1 for e in RSE if e > TAU_MOTION]) / np.count_nonzero(mask)

		return PEPN

	else:
		print("Non-valid error measure. Please, select one of the following: 'EPE', 'MSEN', 'PEPN'")
		return None


"""
	Compute metrics
"""

def test_of_metrics():
	# Path to data
	frame_1 = 'data/seq157/000157_10.png'
	frame_2 = 'data/seq157/000157_11.png'
	flow_noc_path = 'data/seq157/gt/noc/000157_10.png'  # only non-occluded pixels
	flow_val_path = 'data/seq157/gt/occ/000157_10.png'  # ALL valid (non-occ+occluded) pixels

	# Ensure images are grayscale
	f1_gray = cv2.cvtColor(cv2.imread(frame_1), cv2.COLOR_BGR2GRAY)
	f2_gray = cv2.cvtColor(cv2.imread(frame_2), cv2.COLOR_BGR2GRAY)

	# Load the optical flow estimated via Lucas-Kanade
	flow_est_path = 'data/seq157/LKflow_000157_10.png'
	flow_est = readFlow(flow_est_path)


	# Read flows in KITTI format
	flow_gt_noc = read_flow(flow_noc_path)
	gt_u_noc = flow_gt_noc[:, :, 0]	
	gt_v_noc = flow_gt_noc[:, :, 1]
	noc_mask = flow_gt_noc[:, :, 2].astype(np.uint64)  # non-occluded pixels have value 1

	flow_gt_val = read_flow(flow_val_path)
	gt_u_val = flow_gt_val[:, :, 0]	
	gt_v_val = flow_gt_val[:, :, 1]
	val_mask = flow_gt_val[:, :, 2].astype(np.uint64)  # ALL valid pixels (occ+non-occ) have value 1, invalid to 0
	
	# Compute error metrics
	# EPEall
	EPE_all = flow_error(gt_u_val, gt_v_val, u, v, val_mask, 0, method='EPE')  # don't count mask == 0

	# EPEmat
	EPE_mat = flow_error(gt_u_noc, gt_v_noc, u, v, noc_mask, 0, method='EPE')

	# EPEumat
	# A "little" bit trickier. The occluded pixels have value 0 in noc_mask and 1 in occ_mask
	# Use logical operators with masks
	occ_mask = ~noc_mask & val_mask  # i.e.: occluded = not in non_occluded but are valid
	EPE_umat = flow_error(gt_u_val, gt_v_val, u, v, occ_mask, 0, method='EPE')	

	# MSEN
	MSEN = flow_error(gt_u_noc, gt_v_noc, u, v, noc_mask, 0, method='MSEN')

	# PEPN
	MSEN = flow_error(gt_u_noc, gt_v_noc, u, v, noc_mask, 0, method='PEPN')

	# Print metrics
	print("Computed metrics:")
	print("EPEall={}\t EPEmat={}\t EPEumat={}\nMSEN={}\t PEPN={}%".format(EPE_all, EPE_mat, EPE_umat,
																		MSEN, 100.0*PEPN))
	
