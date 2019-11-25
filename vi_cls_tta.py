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

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

cls_name = 'resnext50_32x4d_1115-1540_all'
if 'wide_resnet50_2' in cls_name:
    model_name = 'wide_resnet50_2'
if 'resnext50_32x4d_1115' in cls_name:
    model_name = 'resnext50_32x4d'
else:
    model_name = cls_name.split('_')[0]
# encoder_weights = 'imagenet'
save_dir = os.path.join(SAVE_PATH,'classify',cls_name)
print(cls_name)

NUM_WORKERS = 4
BATCH_SIZE = 32
RESIZE = (480,640)

train_csv = os.path.join(DATA_PATH,'train.csv')
img_path = os.path.join(DATA_PATH,'train_images')
kfold_path = 'kfold.pkl'
data_path = '/home/jianglb/pythonproject/cloud_segment/data/train_{}_{}'.format(*RESIZE)
# data_path = '/home/noel/pythonproject/cloud_segment/data/train_{}_{}'.format(*RESIZE)
test_data = os.path.join(DATA_PATH,'test_{}_{}'.format(*RESIZE))


cls_probs = []
cloud_class = 0

with torch.no_grad():
    for fold in range(K):
        print('Fold{}:'.format(fold))
        cls_model = load_model(model_name,classes=4,dropout=0.,pretrained=False)
        cls_model.load_state_dict(torch.load(os.path.join(save_dir, 'model_{}.pth'.format(fold))))
        cls_model.cuda()
        cls_model.eval()

        preprocessing_fn = smp.encoders.get_preprocessing_fn('resnet34', 'imagenet')

        # validate
        cls_probs_fold = 0
        for tt in range(4):
            validate_dataset = CloudTrainDataset2(train_csv, data_path, kfold_path, fold, phase='validate', transform_type=tt, preprocessing=utils.get_preprocessing(preprocessing_fn),)
            validate_dataloader = DataLoader(validate_dataset,batch_size=BATCH_SIZE,num_workers=NUM_WORKERS,shuffle=False)
            cls_probs_fold_type = []
            for i, (_, imgs, masks, classes) in enumerate(tqdm(validate_dataloader)):
                imgs = imgs.float().cuda()
                predclasses = cls_model.predict(imgs)
                cls_probs_fold_type.append(predclasses.cpu().numpy())

            cls_probs_fold_type = np.concatenate(cls_probs_fold_type)
            cls_probs_fold += cls_probs_fold_type

        cls_probs_fold /= 4
        cls_probs.append(cls_probs_fold)

        del cls_probs_fold
        del cls_probs_fold_type
        gc.collect()

        # inference
        for tt in range(4):
            test_dataset = CloudTestDataset(test_data, transform_type=tt, preprocessing=utils.get_preprocessing(preprocessing_fn), )
            test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

            class_ = []

            for img_names, imgs, _ in tqdm(test_dataloader):
                imgs = imgs.float().cuda()
                predclasses = cls_model.predict(imgs)
                class_.append(predclasses.cpu().numpy())

            class_ = np.concatenate(class_,axis=0)
            cloud_class += class_

        del class_
        gc.collect()

cls_probs = np.concatenate(cls_probs,axis=0)
cloud_class /= (K * 4)

del validate_dataset
del validate_dataloader
del imgs
del classes
del predclasses
del cls_model

del test_dataset
del test_dataloader

gc.collect()

np.save(os.path.join(save_dir,'cls_probs.npy'),cls_probs)
np.save(os.path.join(save_dir,'cloud_class.npy'),cloud_class)




