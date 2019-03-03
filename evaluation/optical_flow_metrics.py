# Optical flow metrics
# Based on KITTI 2012-2015 metrics
# 1. MSEN: Mean Square Error in Non-occluded areas
# 2. PEPN: Percentage of Erroneous Pixels in Non-occluded areas

# To read/write flo files:
# We follow KITTI format which stores the flow fields in a uint16 png file with 3 channels.
# The first 2 contain flow-fields (horizontal + vertical) and the third details invalid/valid pixels (0/1)
# Alternatively, we could have used Middlebury's files 'writeFlow.py', 'readFlow.py' (adapted from Matlab source)
# Which separates the valid mask in a png image instead of encoding everything into one image
# TODO: add maximum and minimum values in absolute value to avoid NaNs or Infs! (e.g.: >=0, <= 1e9)

import numpy as np

tau_motion = 3  # errors smaller or equal to 3 pixels are not taken into account


# 'of_estim': estimated optical flow field (width x height x 2)
# 'of_gt': ground truth optical flow field (width x height x 2)
def SE(F_est, F_gt):  # Squared error
	# Decompose F_gt in horizontal flow + vertical " + valid mask
	F_gt_du = np.squeeze(F_gt[:,:,0])
	F_gt_dv = np.squeeze(F_gt[:,:,1])
	F_gt_val = np.squeeze(F_gt[:,:,2])

	# Decompose F_est in horizontal + vertical flow
	F_est_du = np.squeeze(F_est[:,:,0])
	F_est_dv = np.squeeze(F_est[:,:,1])
	
	# Differences in horizontal and vertical
	E_du = F_gt_du - F_est_du
	E_dv = F_est_dv - F_est_dv
	# Add together
	SE = E_du**2 + E_dv**2

	return SE

def MSEN(F_est, F_gt, occ_mask):  # does not mean EPE, right? (missing sqrt)
	SE = SE(F_est,F_gt)
	
	# Compute mean    
	MSE = np.mean(SE)

	# Exclude non-valid and occluded pixels
	MSE[F_gt_val==0] = 0
	MSE[occ_mask==1] = 0

	MSEN = MSE

    return MSEN


def PEPN(F_est, F_gt, occ_mask):  # computes length(F_err)/length(F_valid)
	SE = SE(F_est,F_gt)
	RSE = np.sqrt(SE)
	# Exclude non-valid and occluded pixels
	RSE[F_gt_val==0] = 0
	RSE[occ_mask==1] = 0
	
	PEPN = len([1 for e in RSE if e > tau_motion]) / np.count_nonzero(F_gt_val)

	return PEPN


def EPE(F_est, F_gt, occ_mask):  # Computes the Endpoint Error (euclidean norm ||F_est-F_gt||)
	SE = SE(F_est, F_gt)
	EPE = np.sqrt(SE)
	MEPE = np.mean(EPE)
	
	# Return EPEall, EPEmat (matched/non-occluded) and EPEumat (unmatched/occluded) (as a tuple)
	# Exclude non-valid and occluded pixels
	MEPE[F_gt_val==0] = 0
	aux = MEPE
	MEPE_all = aux		
	# Non-occluded	
	MEPE[occ_mask==1] = 0
	MEPE_mat = MEPE	
	# Occluded		
	aux[occ_mask==0] = 0
	MEPE_umat = aux	

	return (MEPE, MEPE_mat, MEPE_umat)


