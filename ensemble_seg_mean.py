import numpy as np
import os
import time
import pickle
import gc
from tqdm import tqdm

from global_parameter import *


# candidates = [
#     'FPN_resnext101_32x8d_1107-2310-all',
#     'FPN_efficientnet-b3_1109-0848-all',
#     'FPN_efficientnet-b3_1118-0057-all',
#     'FPN_efficientnet-b3_1104-1734-all',
#     # 'FPN_efficientnet-b0_1113-2136-all',
#     'FPN_resnet34_1110-1049-all',
#     'FPN_resnet34_1113-1740-all',
#     'FPN_resnet50_1110-1640-all',
#     'FPN_resnet101_1110-1041-all',
#     # 'FPN_densenet161_1110-2050-all',
#     # 'FPN_densenet169_1110-2222-all',
#     'FPN_densenet201_1112-1425-all',
#     'FPN_dpn92_1111-2051-all',
#     'FPN_dpn68b_1111-2353-all',
#     # 'FPN_se_resnet50_1114-1138-all',
#     # 'UNET_efficientnet-b3_1105-0551-all',
#     'UNET_resnet34_1111-0558-all',
#     # 'UNET_resnext101_32x8d_1113-0102-all',
#     # 'UNET_densenet161_1115-2032-all',
#     # 'UNET_dpn92_1116-1102-all',
#
#     # 'FPN_resnext101_32x8d_1117-1534-all',
#     # 'FPN_resnet34_1116-2302-all',
#     'FPN_dpn92_1117-0102-all',
#     'FPN_densenet161_1117-0914-all',
# ]

candidates = [
    'Ensemble14_max_1118-1706',
    'Ensemble14_mean_1118-1530'
]

candidates_num = len(candidates)

time_str = time.strftime("%m%d-%H%M", time.localtime())
save_dir = 'Ensemble{}_mean_{}'.format(len(candidates),time_str)
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

weight = 1 / candidates_num


print('Ensembling cloud masks ...')
cloud_masks = 0.
for c in tqdm(candidates):
    cm = np.load(os.path.join(SAVE_PATH,'segment',c,'cloud_mask.npy')).astype(np.float16)
    cloud_masks += (cm * weight)

cloud_masks = np.around(cloud_masks).astype(np.uint8)
print(cloud_masks.min(),cloud_masks.max())
np.save(os.path.join(save_dir,'cloud_mask.npy'),cloud_masks)

del cloud_masks
del cm
gc.collect()

print('Ensembling seg probs ...')
seg_probs = 0.
for c in tqdm(candidates):
    sp = np.load(os.path.join(SAVE_PATH,'segment',c,'seg_probs.npy')).astype(np.float16)
    seg_probs += (sp * weight)
seg_probs = np.around(seg_probs).astype(np.uint8)
print(seg_probs.min(),seg_probs.max())
np.save(os.path.join(save_dir,'seg_probs.npy'),seg_probs)
