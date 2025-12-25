import os
import torch
import random
import numpy as np

from operator import itemgetter

def setRandomSeed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True

def printDash(num = 50):
    print(''.join(['-']*num))

_center_weight = np.load('lib/data/center_weight_origin.npy').reshape(48,48)

def maxPoint(heatmap, center=True):

    if len(heatmap.shape) == 3:
        batch_size,h,w = heatmap.shape
        c = 1

    elif len(heatmap.shape) == 4:
        batch_size,c,h,w = heatmap.shape

    if center:
        heatmap = heatmap*_center_weight

    heatmap = heatmap.reshape((batch_size,c, -1))
    max_id = np.argmax(heatmap,2)
    y = max_id//w
    x = max_id%w
    return x,y

def extract_keypoints(heatmap):

    heatmap = heatmap[0]

    heatmap[heatmap < 0.1] = 0
    heatmap_with_borders = np.pad(heatmap, [(2, 2), (2, 2)], mode='constant')
    heatmap_center = heatmap_with_borders[1:heatmap_with_borders.shape[0]-1, 1:heatmap_with_borders.shape[1]-1]
    heatmap_left = heatmap_with_borders[1:heatmap_with_borders.shape[0]-1, 2:heatmap_with_borders.shape[1]]
    heatmap_right = heatmap_with_borders[1:heatmap_with_borders.shape[0]-1, 0:heatmap_with_borders.shape[1]-2]
    heatmap_up = heatmap_with_borders[2:heatmap_with_borders.shape[0], 1:heatmap_with_borders.shape[1]-1]
    heatmap_down = heatmap_with_borders[0:heatmap_with_borders.shape[0]-2, 1:heatmap_with_borders.shape[1]-1]
    heatmap_peaks = (heatmap_center > heatmap_left) &\
                    (heatmap_center > heatmap_right) &\
                    (heatmap_center > heatmap_up) &\
                    (heatmap_center > heatmap_down)
    heatmap_peaks = heatmap_peaks[1:heatmap_center.shape[0]-1, 1:heatmap_center.shape[1]-1]
    keypoints = list(zip(np.nonzero(heatmap_peaks)[1], np.nonzero(heatmap_peaks)[0]))  # (w, h)
    keypoints = sorted(keypoints, key=lambda x:(x[0]-23.5)**2+(x[1]-23.5)**2)

    suppressed = np.zeros(len(keypoints), np.uint8)

    keypoints_with_score_and_id = []
    for i in range(len(keypoints)):
        if suppressed[i]:
            continue
        for j in range(i+1, len(keypoints)):
            if ((keypoints[i][0] - keypoints[j][0]) ** 2 +
                         (keypoints[i][1] - keypoints[j][1]) ** 2)**0.5 < 6:
                suppressed[j] = 1
        keypoint_with_score_and_id = [keypoints[i][0], keypoints[i][1],
                                    heatmap[keypoints[i][1], keypoints[i][0]],
                                    _center_weight[keypoints[i][1], keypoints[i][0]],
                                    heatmap[keypoints[i][1], keypoints[i][0]]*_center_weight[keypoints[i][1], keypoints[i][0]]]
        keypoints_with_score_and_id.append(keypoint_with_score_and_id)

    keypoints_with_score_and_id = sorted(keypoints_with_score_and_id,
                            key=lambda x:x[-1], reverse=True)
    x,y = keypoints_with_score_and_id[0][0],keypoints_with_score_and_id[0][1]
    return x,y

import numpy as np

def getDist(pre, labels):
        pre = pre.reshape([-1, 17, 2])
        labels = labels.reshape([-1, 17, 2])
        res = np.power(pre[:,:,0]-labels[:,:,0],2)+np.power(pre[:,:,1]-labels[:,:,1],2)
        return res

def getAccRight(dist, img_size, th=None):
        if th is None:
                th = 5 / img_size
        res = np.zeros(dist.shape[1], dtype=np.int64)
        for i in range(dist.shape[1]):
                res[i] = sum(dist[:,i]<th)

        return res

def myAcc(output, target, img_size, th=None, kps_mask=None, kps_th=0.0):
    if len(output.shape) == 4:
            output = heatmap2locate(output)
            target = heatmap2locate(target)

    dist = getDist(output, target)
    if th is None:
            th = 5 / img_size

    if kps_mask is not None:
            if torch.is_tensor(kps_mask):
                    kps_mask = kps_mask.detach().cpu().numpy()
            mask = (kps_mask > kps_th)
            if mask.shape != dist.shape:
                    mask = mask.reshape(dist.shape)
            correct = np.sum((dist < th) & mask, axis=0).astype(np.int64)
            total = np.sum(mask, axis=0).astype(np.int64)
            return correct, total

    correct = np.sum(dist < th, axis=0).astype(np.int64)
    total = np.full(dist.shape[1], dist.shape[0], dtype=np.int64)
    return correct, total
