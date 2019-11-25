import numpy as np
import os
import time
import pickle
import gc
from tqdm import tqdm

from global_parameter import *

candidates = [
    'FPN_resnext101_32x8d_1107-2310-all',
    'FPN_efficientnet-b3_1109-0848-all',
    'FPN_efficientnet-b3_1118-0057-all',
    'FPN_efficientnet-b3_1104-1734-all',
    # 'FPN_efficientnet-b0_1113-2136-all',
    'FPN_resnet34_1110-1049-all',
    'FPN_resnet34_1113-1740-all',
    'FPN_resnet50_1110-1640-all',
    'FPN_resnet101_1110-1041-all',
    # 'FPN_densenet161_1110-2050-all',
    # 'FPN_densenet169_1110-2222-all',
    'FPN_densenet201_1112-1425-all',
    'FPN_dpn92_1111-2051-all',
    'FPN_dpn68b_1111-2353-all',
    # 'FPN_se_resnet50_1114-1138-all',
    # 'UNET_efficientnet-b3_1105-0551-all',
    'UNET_resnet34_1111-0558-all',
    # 'UNET_resnext101_32x8d_1113-0102-all',
    # 'UNET_densenet161_1115-2032-all',
    # 'UNET_dpn92_1116-1102-all',

    # 'FPN_resnext101_32x8d_1117-1534-all',
    # 'FPN_resnet34_1116-2302-all',
    'FPN_dpn92_1117-0102-all',
    'FPN_densenet161_1117-0914-all',
]

method = 'max'
time_str = time.strftime("%m%d-%H%M", time.localtime())
save_dir = 'Ensemble{}_{}_{}'.format(len(candidates),method, time_str)
save_dir = os.path.join(SAVE_PATH,'segment',save_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
with open(os.path.join(save_dir,'config'), 'w') as f:
    f.write('{}\n\n'.format(time_str,))
    for c in candidates:
        f.write('{}\n'.format(c))
with open(os.path.join(save_dir,'candidates.pkl'),'wb') as f:
    pickle.dump(candidates,f)
print(save_dir)


print('Ensembling cloud masks ...')
if method == 'max':
    cloud_masks = np.zeros((TEST_NUM,4,*RESIZE),dtype=np.uint8)
else:
    cloud_masks = np.ones((TEST_NUM,4,*RESIZE),dtype=np.uint8) * 110

for c in tqdm(candidates):
    cm = np.load(os.path.join(SAVE_PATH,'segment',c,'cloud_mask.npy')).astype(np.uint8)
    if method == 'max':
        cloud_masks = np.maximum(cloud_masks,cm)
    else:
        cloud_masks = np.minimum(cloud_masks,cm)

print(cloud_masks.min(),cloud_masks.max())
np.save(os.path.join(save_dir,'cloud_mask.npy'),cloud_masks)

del cloud_masks
del cm
gc.collect()

print('Ensembling seg probs ...')
if method == 'max':
    seg_probs = np.zeros((TRAIN_NUM,4,*RESIZE),dtype=np.uint8)
else:
    seg_probs = np.ones((TRAIN_NUM,4,*RESIZE),dtype=np.uint8) * 110

for c in tqdm(candidates):
    sp = np.load(os.path.join(SAVE_PATH,'segment',c,'seg_probs.npy')).astype(np.uint8)
    if method == 'max':
        seg_probs = np.maximum(seg_probs,sp)
    else:
        seg_probs = np.minimum(seg_probs,sp)

print(seg_probs.min(),seg_probs.max())
np.save(os.path.join(save_dir,'seg_probs.npy'),seg_probs)
