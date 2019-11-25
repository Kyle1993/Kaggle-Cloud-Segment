import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import gc
import time

import torch
from torch.utils.data import DataLoader

from global_parameter import *
from dataset import CloudTrainDataset2,CloudTestDataset
import segmentation_models_pytorch as smp
import utils

import warnings

warnings.filterwarnings('ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

seg_name = 'FPN_efficientnet-b3_1118-0057-all'
save_dir = os.path.join(SAVE_PATH,'segment',seg_name)
print(seg_name)

with open(os.path.join(save_dir,'config.pkl'),'rb') as f:
    config = pickle.load(f)

locals().update(config)
NUM_WORKERS = 4
BATCH_SIZE = 32

train_csv = os.path.join(DATA_PATH,'train.csv')
img_path = os.path.join(DATA_PATH,'train_images')
kfold_path = 'kfold.pkl'
data_path = '/home/jianglb/pythonproject/cloud_segment/data/train_{}_{}'.format(*RESIZE)
test_data = os.path.join(DATA_PATH,'test_{}_{}'.format(*RESIZE))

seg_probs = []
cloud_mask = 0

with torch.no_grad():
    for fold in range(K):
        print('Fold{}:'.format(fold))
        if MODEL == 'UNET':
            seg_model = smp.Unet(encoder_name=ENCODER,classes=4,encoder_weights=None,ED_drop=ED_drop)
        elif MODEL == 'FPN':
            seg_model = smp.FPN(encoder_name=ENCODER,classes=4,encoder_weights=None,ED_drop=ED_drop,dropout=dropout)
        else:
            seg_model = None
        seg_model.load_state_dict(torch.load(os.path.join(save_dir, 'model_{}.pth'.format(fold))))
        seg_model.cuda()
        seg_model.eval()

        preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, encoder_weights)

        # validate
        seg_probs_fold = 0
        for tt in range(4):
            validate_dataset = CloudTrainDataset2(train_csv, data_path, kfold_path, fold, phase='validate', transform_type=tt, preprocessing=utils.get_preprocessing(preprocessing_fn),)
            validate_dataloader = DataLoader(validate_dataset,batch_size=BATCH_SIZE,num_workers=NUM_WORKERS,shuffle=False)
            seg_probs_fold_type = []
            cls_probs_fold_type = []
            for i, (_, imgs, masks, classes) in enumerate(tqdm(validate_dataloader)):
                imgs = imgs.float().cuda()
                predmasks = seg_model.predict(imgs)

                seg_probs_fold_type.append(predmasks.cpu().numpy().astype(np.float16))
            seg_probs_fold_type = np.concatenate(seg_probs_fold_type,axis=0).astype(np.float16)

            if tt == 0:
                pass
            elif tt == 1:
                seg_probs_fold_type = np.flip(seg_probs_fold_type,3)
            elif tt == 2:
                seg_probs_fold_type = np.flip(seg_probs_fold_type,2)
            else:
                seg_probs_fold_type = np.flip(seg_probs_fold_type,2)
                seg_probs_fold_type = np.flip(seg_probs_fold_type,3)

            seg_probs_fold += seg_probs_fold_type

        seg_probs_fold /= 4
        seg_probs_fold = np.around(seg_probs_fold * 100).astype(np.uint8)

        seg_probs.append(seg_probs_fold)

        del seg_probs_fold
        del seg_probs_fold_type
        gc.collect()

        # inference
        for tt in range(4):
            test_dataset = CloudTestDataset(test_data, transform_type=tt, preprocessing=utils.get_preprocessing(preprocessing_fn), )
            test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

            mask_ = []

            for img_names, imgs, _ in tqdm(test_dataloader):
                imgs = imgs.float().cuda()
                predmasks = seg_model.predict(imgs)

                mask_.append(predmasks.cpu().numpy().astype(np.float16))

            mask_ = np.concatenate(mask_,axis=0).astype(np.float16)

            if tt == 0:
                pass
            elif tt == 1:
                mask_ = np.flip(mask_,3)
            elif tt == 2:
                mask_ = np.flip(mask_,2)
            else:
                mask_ = np.flip(mask_,2)
                mask_ = np.flip(mask_,3)

            cloud_mask += mask_

            del mask_
            gc.collect()

seg_probs = np.concatenate(seg_probs,axis=0).astype(np.uint8)

cloud_mask /= (K * 4)
cloud_mask = np.around(cloud_mask * 100).astype(np.uint8)

del validate_dataset
del validate_dataloader
del imgs
del masks
del predmasks
del seg_model

del test_dataset
del test_dataloader

gc.collect()

np.save(os.path.join(save_dir,'seg_probs.npy'),seg_probs)
np.save(os.path.join(save_dir,'cloud_mask.npy'),cloud_mask)

print(seg_name)





