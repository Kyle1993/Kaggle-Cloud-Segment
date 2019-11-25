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
from models import load_model

import warnings

warnings.filterwarnings('ignore')

train_csv = os.path.join(DATA_PATH,'train.csv')
kfold_path = 'kfold.pkl'
data_path = '/home/jianglb/pythonproject/cloud_segment/data/train_{}_{}'.format(*RESIZE)
test_data = os.path.join(DATA_PATH,'test_{}_{}'.format(*RESIZE))
encoder_weights = 'instagram'

tg_masks = []
tg_classes = []
tg_names = []

with torch.no_grad():
    for fold in range(K):
        # validate
        preprocessing_fn = smp.encoders.get_preprocessing_fn('resnext101_32x8d', encoder_weights)
        validate_dataset = CloudTrainDataset2(train_csv, data_path, kfold_path, fold, phase='validate', transform_type=0, preprocessing=utils.get_preprocessing(preprocessing_fn),)
        validate_dataloader = DataLoader(validate_dataset,batch_size=BATCH_SIZE,num_workers=4,shuffle=False)

        for i, (names, imgs, masks, classes) in enumerate(tqdm(validate_dataloader)):

            tg_masks.append(np.around(masks.numpy() * 100).astype(np.uint8))
            tg_classes.append(classes.numpy())
            tg_names.extend(names)


# tg_masks = np.concatenate(tg_masks,axis=0).astype(np.uint8)
# tg_classes = np.concatenate(tg_classes,axis=0)

# print(cloud_mask.max(),cloud_mask.min())

del validate_dataset
del validate_dataloader
del imgs
del masks
del classes

gc.collect()


# np.save('tg_masks_.npy',tg_masks)
# np.save('tg_classes_.npy',tg_classes)
with open('tg_names.pkl','wb') as f:
    pickle.dump(tg_names,f)





