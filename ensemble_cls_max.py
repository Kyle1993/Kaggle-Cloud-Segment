import numpy as np
import os
import time
import pickle
import gc
from tqdm import tqdm
import random

from global_parameter import *

def softmax(w):
    w_ = np.asarray(w)
    return np.exp(w_)/np.sum(np.exp(w_))

candidates = [
    'wide_resnet50_2_1115-0846_all',
    'resnet50_1115-1052_all',
    'resnet101_1115-0130_all',
    'densenet161_1115-0140_all',
    'resnext50_32x4d_1115-1540_all',
      ]

candidates_num = len(candidates)

method = 'max'

time_str = time.strftime("%m%d-%H%M", time.localtime())
save_dir = 'Ensemble{}_{}_{}'.format(len(candidates), method, time_str)
save_dir = os.path.join(SAVE_PATH,'classify',save_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
print(save_dir)

if method == 'max':
    cloud_class = np.zeros((TEST_NUM,4))
    cls_probs = np.zeros((TRAIN_NUM,4))
else:
    cloud_class = np.ones((TEST_NUM,4))
    cls_probs = np.ones((TRAIN_NUM,4))

for c in candidates:
    cc = np.load(os.path.join(SAVE_PATH,'classify',c,'cloud_class.npy'))
    cp = np.load(os.path.join(SAVE_PATH,'classify',c,'cls_probs.npy'))
    if method == 'max':
        cloud_class = np.maximum(cloud_class, cc)
        cls_probs = np.maximum(cls_probs, cp)
    else:
        cloud_class = np.minimum(cloud_class, cc)
        cls_probs = np.minimum(cls_probs, cp)

print(cloud_class.shape,cloud_class.min(),cloud_class.max())
print(cls_probs.shape,cls_probs.min(),cls_probs.max())

np.save(os.path.join(save_dir,'cloud_class.npy'),cloud_class)
np.save(os.path.join(save_dir,'cls_probs.npy'),cls_probs)
