import numpy as np
import pickle
import os
import pandas as pd
import gc
import time

import utils
from global_parameter import *

base_folder = 'submits'
seg_fold = 'FPN_resnet34_1116-2302-all'
cls_fold = 'resnet50_1105-1539_all'

print(seg_fold)
with open(os.path.join(SAVE_PATH,'segment',seg_fold,'config.pkl'),'rb') as f:
    config = pickle.load(f)
encoder_weights = config['encoder_weights']
seg_name = '{}{}'.format(config['MODEL'],config['ENCODER'])
cls_name = cls_fold.split('_')[0]
time_str = time.strftime("%m%d-%H%M", time.localtime())

save_dir = os.path.join(base_folder,'{}_{}_{}'.format(seg_name,cls_name,time_str))

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
with open(os.path.join(save_dir,'config'), 'w') as f:
    f.write('{}\n\n{}'.format(time_str, utils.config2str({'seg_fold':seg_fold, 'cls_fold':cls_fold})))

# search thresholds
seg_probs = np.load(os.path.join(SAVE_PATH,'segment',seg_fold,'seg_probs.npy'))
cls_probs = np.load(os.path.join(SAVE_PATH,'classify',cls_fold,'cls_probs.npy'))

tg_masks = np.load(os.path.join(SAVE_PATH,'tg','tg_masks.npy'))
tg_classes = np.load(os.path.join(SAVE_PATH,'tg','tg_classes.npy'))


cls_threshold, seg_threshold, cpn_threshold, best_dice = utils.search_threshold(tg_masks,seg_probs,tg_classes,cls_probs)

# cls_threshold = [0.4, 0.4, 0.4, 0.4]
# seg_threshold = [0.5, 0.5, 0.5, 0.5]
# cpn_threshold = [25000, 20000, 25000, 10000]
# best_dice = [0.6208261045509903, 0.7649952469320047, 0.6188291453734969, 0.6120651802644084]

with open(os.path.join(save_dir,'config'),'a') as f:
    f.write('\n\ncls_threshold:{}\nseg_threshold:{}\ncpn_threshold:{}\nbest dice:{}\ncv dice:{}'.format(cls_threshold,seg_threshold,cpn_threshold,best_dice,np.mean(best_dice)))

del seg_probs
del cls_probs
del tg_masks
del tg_classes
gc.collect()

# generate submission
print('generating submission.csv...')
cloud_mask = np.load(os.path.join(SAVE_PATH,'segment',seg_fold,'cloud_mask.npy'))
cloud_class = np.load(os.path.join(SAVE_PATH,'classify',cls_fold,'cloud_class.npy'))
with open(os.path.join(SAVE_PATH,'tg','names.pkl'),'rb') as f:
    names = pickle.load(f)

cloud_class_maxi =cloud_class.argmax(axis=1)

m2r = utils.Mask2Rle(LABELS,resize=STANDARD_RESIZE,seg_threshold=seg_threshold,cpn_threshold=cpn_threshold)

for i in range(4):
    cloud_class[:, i][cloud_class[:, i]<=cls_threshold[i]] = 0
    cloud_class[:, i][cloud_class[:, i]>cls_threshold[i]] = 1

# print((cloud_class.sum(axis=1) == 0).sum())
# # force at least one positive predict
# neg = (cloud_class.sum(axis=1) == 0)
# for i,v in enumerate(neg):
#     if v:
#         cloud_class[i,cloud_class_maxi[i]] = 1
# print((cloud_class.sum(axis=1) == 0).sum())

cloud_class = np.tile(cloud_class.reshape((*cloud_class.shape,1,1)),(1,1,cloud_mask.shape[-2],cloud_mask.shape[-1]))
cloud_mask = cloud_mask * cloud_class

Image_Label = []
EncodedPixels = []
names, pixels = m2r.batch_convert(names,cloud_mask,cloud_class_maxi)
Image_Label.extend(names)
EncodedPixels.extend(pixels)

submit_df = pd.DataFrame({'Image_Label':Image_Label, 'EncodedPixels':EncodedPixels})
submit_df.set_index('Image_Label',inplace=True)
submit_df.to_csv(os.path.join(save_dir, 'submission_{}.csv'.format(time_str)))

print('Done')
print('Save in:', save_dir)

