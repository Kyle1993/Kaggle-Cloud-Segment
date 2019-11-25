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
      ]

candidates_num = len(candidates)

time_str = time.strftime("%m%d-%H%M", time.localtime())
save_dir = 'Ensemble{}_{}'.format(len(candidates),time_str)
save_dir = os.path.join(SAVE_PATH,'classify',save_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
print(save_dir)

# calculate weight
method = 'average' # average, weighted

with open(os.path.join(save_dir,'config'), 'w') as f:
    f.write('{}\n\n'.format(time_str,))
    f.write('method:{}\n\n'.format(method))

tg_classes = np.load(os.path.join(SAVE_PATH,'tg','tg_classes.npy'))

cloud_classes = []
cloud_probs = []
for c in candidates:
    cc = np.load(os.path.join(SAVE_PATH,'classify',c,'cloud_class.npy'))
    cp = np.load(os.path.join(SAVE_PATH,'classify',c,'cls_probs.npy'))
    cloud_classes.append(cc)
    cloud_probs.append(cp)
cloud_classes = np.asarray(cloud_classes)
cloud_probs = np.asarray(cloud_probs)

assert cloud_probs.shape[-2:] == tg_classes.shape
assert cloud_probs.shape[0] == cloud_classes.shape[0]


# weighted
if method == 'weighted':
    weights = []
    wait_round = 100
    for i in range(4):
        w = np.zeros((candidates_num,))
        tg = tg_classes[:,i]
        cp = cloud_probs[:,:,i]

        best_score = np.inf
        p = np.zeros((tg.shape[0],))
        counter = 0
        while counter < wait_round:
            sample = random.sample(list(range(candidates_num)),2)

            sp1 = cp[sample[0]]
            sp2 = cp[sample[1]]

            p1 = (p * w.sum() + sp1) / (w.sum() + 1)
            p2 = (p * w.sum() + sp2) / (w.sum() + 1)
            p3 = (p * w.sum() + sp1 + sp2) / (w.sum() + 2)
            # print(p1.mean(),p2.mean(),p3.mean())

            s1 = np.sum(np.abs(p1 - tg)**2)
            s2 = np.sum(np.abs(p2 - tg)**2)
            s3 = np.sum(np.abs(p3 - tg)**2)
            # print(s1,s2,s3)
            min_case = np.argmin([best_score,s1,s2,s3])
            # print(min_case)

            if min_case == 0:
                counter += 1
            elif min_case == 1:
                best_score = s1
                p = p1
                w[sample[0]] += 1
                counter = 0
            elif min_case == 2:
                best_score = s2
                p = p2
                w[sample[1]] += 1
                counter = 0
            else:
                best_score = s3
                p = p3
                w[sample[0]] += 1
                w[sample[1]] += 1
                counter = 0
            # print('{}\t{}\t{}'.format(i, best_score, w))
        # exit()
        print('{}\t{}\t{}'.format(i,best_score,w))
        weights.append(w / w.sum())

    weights = np.asarray(weights)

# # average
elif method == 'average':
    weights = np.ones((4,candidates_num))
    weights /= candidates_num

else:
    raise Exception('Error method')

print(weights)

with open(os.path.join(save_dir,'config'),'a') as f:
    for i,c in enumerate(candidates):
        f.write('{}:\t{}\n'.format(c,'[{:.4f}, {:.4f}, {:.4f}, {:.4f}]'.format(weights[0,i],weights[1,i],weights[2,i],weights[3,i])))
        print('{}:\t{}'.format(c,'[{:.4f}, {:.4f}, {:.4f}, {:.4f}]'.format(weights[0,i],weights[1,i],weights[2,i],weights[3,i])))

config = {'candidates':candidates,
          'method':method,
          'weights':weights}
with open(os.path.join(save_dir,'config.pkl'),'wb') as f:
    pickle.dump(config,f)

print('Ensembling...')
cloud_class = np.zeros_like(cloud_classes[0])
cloud_prob = np.zeros_like(cloud_probs[0])
for i in range(4):
    cloud_prob[:,i] = np.dot(cloud_probs[:,:,i].T, weights[i].T)
    cloud_class[:,i] = np.dot(cloud_classes[:,:,i].T, weights[i].T)


print(cloud_class.shape,cloud_class.min(),cloud_class.max())
print(cloud_prob.shape,cloud_prob.min(),cloud_prob.max())

np.save(os.path.join(save_dir,'cloud_class.npy'),cloud_class)
np.save(os.path.join(save_dir,'cls_probs.npy'),cloud_prob)
