import numpy as np
from sklearn.metrics import precision_recall_curve


def compute_average_precision(y_true, y_predict, interpolate=True, point_11=False):
    """
    Average precision in different formats: (non-) interpolated and/or 11-point approximated
    point_11=True and interpolate=True corresponds to the 11-point interpolated AP used in
    the PASCAL VOC challenge up to the 2008 edition and has been verfied against the vlfeat implementation
    The exact average precision (interpolate=False, point_11=False) corresponds to the one of vl_feat
    :param y_true: list/ numpy vector of true labels in {0,1} for each element
    :param y_predict: predicted score for each element
    :param interpolate: Use interpolation?
    :param point_11: Use 11-point approximation to average precision?
    :return: average precision
    """
    # Check inputs
    assert len(y_true)==len(y_predict), "Prediction and ground truth need to be of the same length"
    if len(set(y_true))==1:
        if y_true[0]==0:
            raise ValueError('True labels cannot all be zero')
        else:
            return 1
    else:
        assert sorted(set(y_true))==[0,1], "Ground truth can only contain elements {0,1}"

    # Compute precision and recall
    precision, recall, _ = precision_recall_curve(y_true, y_predict)
    recall=recall.astype(np.float32)

    if interpolate: # Compute the interpolated precision
        for i in range(1,len(precision)):
            precision[i]=max(precision[i-1],precision[i])

    if point_11: # Compute the 11-point approximated AP
        precision_11=[precision[np.where(recall>=t)[0][-1]] for t in np.arange(0,1.01,0.1)]
        return np.mean(precision_11)
    else: # Compute the AP using precision at every additionally recalled sample
        indices=np.where(np.diff(recall))
        return np.mean(precision[indices])
