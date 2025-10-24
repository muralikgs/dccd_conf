
import numpy as np

def compute_shd(W_gt, W_est):
    
    '''
    Returns the Structural Hamming Distance (SHD) between two graphs
    
    Parameters
    ----------
    W_gt : Ground truth graph in the form of numpy.ndarray, 
           should be binary or bool matrix
    W_est : Estimated matrix in the form of numpy.ndarray,
            should be binary or bool matrix
    '''
    
    # both W_gt & W_est should be binary matrices
    W_gt = W_gt * 1.0
    W_est = W_est * 1.0
    
    corr_edges = (W_gt == W_est) * W_gt # All the correctly identified edges
    
    W_gt -= corr_edges
    W_est -= corr_edges
    
    R = (W_est.T == W_gt) * W_gt # Reverse edges
    
    W_gt -= R
    W_est -= R.T
    
    E = W_est > W_gt # Extra edges
    M = W_est < W_gt # Missing edges

    return R.sum() + E.sum() + M.sum(), (R.sum(), E.sum(), M.sum())

def norm_shd(W_gt, W_est):
    shd, _ = compute_shd(W_gt, W_est)
    return shd / W_gt.shape[0]

def get_error_metrics(W_gt, W_est):
    # Computes the accuracy, precision and recall between the estimated W and true W. 
    # Parameters:
    # 1) W_gt - Ground truth adjacency matrix.
    # 2) W_est - Estimated adjacency matrix.
    # Both the matrices are assumed to binary. 

    W_gt = W_gt * 1.0
    W_est = W_est * 1.0

    acc = (W_gt == W_est).sum()
    tp_matrix = W_gt * W_est
    fp_matrix = (W_est - tp_matrix).sum()
    
    return tp_matrix.sum(), acc - tp_matrix.sum(), fp_matrix.sum(), (W_gt - tp_matrix).sum() 

# def compute_auprc_conf(B_gt, cov_est, n_points, min_threshold=):

#     B_est = 

def compute_auprc(W_gt, W_est, n_points=50, min_threshold=0, logspace=False):

    if logspace:
        max_threshold = np.log(W_est.max())
        threshold_list = np.logspace(min_threshold, max_threshold, n_points)
    else:
        max_threshold = W_est.max()
        threshold_list = np.linspace(min_threshold, max_threshold, n_points)
    
    rec_list, pre_list = list(), list()
    for threshold in threshold_list:
        tp, tn, fp, fn = get_error_metrics(W_gt, W_est >= threshold)
        rec = tp / (tp + fn)
        pre = tp / (tp + fp)
        rec_list.append(rec)
        pre_list.append(pre)

    rec_list.append(0)
    pre_list.append(1.0)

    area = np.trapz(pre_list[::-1], rec_list[::-1])
    baseline = W_gt.sum() / (W_gt.shape[0] * W_gt.shape[1])

    return baseline, area, pre_list, rec_list