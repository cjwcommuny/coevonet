import numpy as np

def IoU(window1: np.ndarray, window2: np.ndarray) -> np.ndarray:
    """
    :param window1: shape=(N, 2)
    :param window2: shape=(N, 2)
    :return IoUs: shape=(N,)
    """
    intersection_size = np.maximum(
        0,
        np.minimum(window1[:,1], window2[:,1]) - np.maximum(window1[:,0], window2[:,0])
    )
    union_size = np.maximum(window1[:,1], window2[:,1]) - np.minimum(window1[:,0], window2[:,0])
    return intersection_size / union_size


def RankIoU(top_n: int, iou_threshold: float, windows: np.ndarray, gt: np.ndarray) -> float:
    """
    compute R@{top_n},IoU={iou_threshold}
    :param top_n:
    :param iou_threshold:
    :param windows: shape=(N, 2)
    :param gt: shape=(2,)
    """
    if len(windows) > top_n:
        windows = windows[:top_n, :]
    gt = np.broadcast_to(gt.reshape((1,2)), windows.shape)
    ious = IoU(windows, gt)
    correct = 1 if np.sum(ious >= iou_threshold) > 0 else 0 # at leat one window satisfy the condition
    return correct
