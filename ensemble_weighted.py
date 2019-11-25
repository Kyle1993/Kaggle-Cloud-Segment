import numpy as np
import os
import time
import pickle
import gc
from tqdm import tqdm

from global_parameter import *

def softmax(w):
    w_ = np.asarray(w)
    return np.exp(w_)/np.sum(np.exp(w_))

score = ['Loss', 'CV', 'LB']
candidates = {
    'FPN_resnext101_32x8d_1107-2310-all': [0.5951, 0.654251748 ,0.6603],
    'FPN_efficientnet-b3_1109-0848-all':[0.5997, None, None],
    'FPN_efficientnet-b0_1113-2136-all':[0.6077, None, None],
    'FPN_resnet34_1110-1049-all':[0.5978, 0.653828776, None],
    # 'FPN_resnet34_1113-1740-all':[0.5988, 0.652718181, 0.6573],
    'FPN_resnet50_1110-1640-all':[0.6045, None, None],
    'FPN_resnet101_1110-1041-all':[0.6014, None, None],
    'FPN_densenet161_1110-2050-all':[0.5983, 0.653520104, None],
    'FPN_densenet169_1110-2222-all':[0.5983, 0.653907754, 0.6625],
    'FPN_densenet201_1112-1425-all':[0.5998, 0.654263626, 0.6611],
    'FPN_dpn92_1111-2051-all':[0.5964, 0.653598043, 0.6624],
    'FPN_dpn68b_1111-2353-all':[0.6008, 0.653101522,0.6594],
    'UNET_efficientnet-b3_1105-0551-all':[0.6026, None, None],
    'UNET_resnet34_1111-0558-all':[0.6062, None, None],
    'UNET_resnext101_32x8d_1113-0102-all':[0.6181, None, None],
      }

time_str = time.strftime("%m%d-%H%M", time.localtime())
save_dir = 'Ensemble{}_{}'.format(len(candidates),time_str)
save_dir = os.path.join(SAVE_PATH,'segment',save_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
with open(os.path.join(save_dir,'config'), 'w') as f:
    f.write('{}\n\n'.format(time_str,))
with open(os.path.join(save_dir,'candidates.pkl'),'wb') as f:
    pickle.dump(candidates,f)
print(save_dir)

# calculate weight
key = 0
method = 'average'

if method == 'softmax':
    weight = []
    cs = []
    for c,v in sorted(candidates.items(),key=lambda x:x[0]):
        w = v[key] if v[key] is not None else -np.inf
        if key == 0:
            w = (1-w) * 100
        else:
            w = w * 1000
        weight.append(w)
        cs.append(c)

    weight = np.asarray(weight)
    weight = softmax(weight)

elif method == 'average':
    weight = []
    cs = []
    for c,v in sorted(candidates.items(),key=lambda x:x[0]):
        files = os.listdir(os.path.join(SAVE_PATH, 'segment', c))
        if 'cloud_mask.npy' in files or 'seg_probs.npy' in files:
            weight.append(1)
        else:
            weight.append(0)
        cs.append(c)
    weight = np.asarray(weight)
    weight = weight / weight.sum()

with open(os.path.join(save_dir,'config'),'a') as f:
    f.write('Key:{}({})\n'.format(score[key],key))
    f.write('Method:{}\n\n'.format(method))

candidate_weight = {}
with open(os.path.join(save_dir,'config'),'a') as f:
    for c,w in zip(cs,weight):
        candidate_weight[c] = w
        f.write('{}:\t{}\n'.format(c,w))
        print('{}:\t{}'.format(c,w))

config = {'key':key,
          'method':method,
          'candidate_weight':candidate_weight}
with open(os.path.join(save_dir,'config.pkl'),'wb') as f:
    pickle.dump(config,f)


print('Ensembling cloud masks ...')
cloud_masks = 0.
for c,w in tqdm(candidate_weight.items()):
    if w != 0.:
        cm = np.load(os.path.join(SAVE_PATH,'segment',c,'cloud_mask.npy')).astype(np.float16)
        cloud_masks += (cm * w)

cloud_masks = np.around(cloud_masks).astype(np.uint8)
print(cloud_masks.min(),cloud_masks.max())
np.save(os.path.join(save_dir,'cloud_mask.npy'),cloud_masks)

del cloud_masks
del cm
gc.collect()

print('Ensembling seg probs ...')
seg_probs = 0.
for c,w in tqdm(candidate_weight.items()):
    if w != 0.:
        sp = np.load(os.path.join(SAVE_PATH,'segment',c,'seg_probs.npy')).astype(np.float16)
        seg_probs += (sp * w)
seg_probs = np.around(seg_probs).astype(np.uint8)
print(seg_probs.min(),seg_probs.max())
np.save(os.path.join(save_dir,'seg_probs.npy'),seg_probs)
