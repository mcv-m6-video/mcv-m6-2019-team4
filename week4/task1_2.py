# Task 1.2 off-the shelf optical flow algorithms
# We asses some optical flow algorithms (MSEN, PEPN)
# Algoritms assessed:
#   * PyFlow (Coarse2Fine OF) [2004]
#      Paper: not clear (is a mixed of at least a couple)
#      [2] 	T. Brox, A. Bruhn, N. Papenberg, and J.Weickert. High accuracy optical flow estimation based on a theory
#       for warping. In European Conference on Computer Vision (ECCV), pages 25–36, 2004.
#      [3] 	A. Bruhn, J.Weickert and C. Schn¨orr. Lucas/Kanade meets Horn/Schunk: combining local and global optical
#       flow methods. International Journal of Computer Vision (IJCV), 61(3):211–231, 2005.
#      Implementation *used*: Python wrapper for Ce Liu's C++ implementation of Coarse2Fine Optical Flow
#           at: https://github.com/pathak22/pyflow
#           Original c++ at: https://people.csail.mit.edu/celiu/OpticalFlow/
#   * EpicFlow [2015]:
#      Paper: https://hal.inria.fr/hal-01142656/document
#      Implementation *used*: https://thoth.inrialpes.fr/src/epicflow/
#   * PWC-Net:
#      Paper:
#      Implementation *used*:
#
import os
from utils import optical_flow as of_utils
from evaluation import optical_flow as of_eval


if __name__ == '__main__':
    path_to_seq = '../../week4_kitti_seq45_offtheshelf'
    # Compute the error for all the OF algorithms tested
    # 1. Load ground-truth as it is common for every algorithm
    path_to_noc = os.path.join(path_to_seq, 'gt_flow/noc/000045_10.png')
    path_to_val = os.path.join(path_to_seq, 'gt_flow/occ/000045_10.png')
    # the nomenclature of calling all valid pixels 'occ' is very confusing... (one reason to rename our variables)
    # in this case exists, we can compute EPE for ONLY occluded pixels as we know ALL valid pixels

    # 2. Load and evaluate each algorithm, printing out the resulting metrics
    # 2.1. PyFlow
    print("Computing results for 'pyflow' (coarse2fine)...")
    print("for grayscale (there is some bug from visual inspection)")
    path_to_pyflow_gray = os.path.join(path_to_seq, 'PyFlow/gray/000045_10_out.npy')
    of_eval.eval_sequence(path_to_noc, path_to_pyflow_gray, path_to_val)

    print("for rgb (works as expected)")
    path_to_pyflow_rgb = os.path.join(path_to_seq, 'PyFlow/rgb/000045_10_out.npy')
    of_eval.eval_sequence(path_to_noc, path_to_pyflow_rgb, path_to_val)

    # 2.2. EpicFlow (only works with rgb)
    print("Computing results for 'EpicFlow'...")
    path_to_epicflow = os.path.join(path_to_seq, 'EpicFlow/ef_seq45.flo')
    of_eval.eval_sequence(path_to_noc, path_to_epicflow, path_to_val)

    # 2.3. PWC-Net (only works with rgb)
    print("Computing results for 'PWC-Net'...")
    path_to_pwcnet = os.path.join(path_to_seq, 'PWC-Net/seq45.flo')
    of_eval.eval_sequence(path_to_noc, path_to_pwcnet, path_to_val)

    # 2.4. FlowNet2 (hopefully)



