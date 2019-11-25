import torch
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from apex import amp

import os
import time
from tqdm import tqdm
import pickle

from global_parameter import *
from dataset import CloudTrainDataset2
import segmentation_models_pytorch as smp
import utils
from loss import FocalBCEDiceLoss
import warnings

warnings.filterwarnings('ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = '0,3'
device_ids = range(torch.cuda.device_count())
use_apex = True

time_str = time.strftime("%m%d-%H%M", time.localtime())

param = {
    'resnet34':{'encoder_weights':'imagenet', 'batch_size':16 if use_apex else 8},
    'se_resnet50': {'encoder_weights': 'imagenet', 'batch_size': 12 if use_apex else 6},
    'dpn92':{'encoder_weights':'imagenet+5k', 'batch_size':8 if use_apex else 4},
    'dpn68b':{'encoder_weights':'imagenet+5k', 'batch_size':12 if use_apex else 6},
    'densenet201':{'encoder_weights':'imagenet', 'batch_size':8 if use_apex else 4},
    'densenet161': {'encoder_weights': 'imagenet', 'batch_size': 8 if use_apex else 4},
    'efficientnet-b0':{'encoder_weights':'imagenet', 'batch_size':12 if use_apex else 6},
    'efficientnet-b3':{'encoder_weights':'imagenet', 'batch_size':8 if use_apex else 4},
    'resnext101_32x8d': {'encoder_weights': 'instagram', 'batch_size': 6 if use_apex else 3},
}

config = {'MODEL':'FPN',
          'ENCODER':'dpn68b', #efficientnet-b3, resnet34, resnext101_32x8d, resnext101_32x16d,
          'NUM_EPOCH':50,
          'RESIZE':(480, 640),
          'LR':5e-4,
          'WD':1e-5,
          'early_stop':5,
          'dropout':0.2,
          'ED_drop':0.2,
          'patience':1,
          'gamma':2,
          'note':'',
          'train_time':time_str,
          }
NUM_WORKERS = len(device_ids) * 4
config['BATCH_SIZE'] = len(device_ids) * param[config['ENCODER']]['batch_size']
config['encoder_weights'] = param[config['ENCODER']]['encoder_weights']

locals().update(config)

train_csv = os.path.join(DATA_PATH,'train.csv')
kfold_path = 'kfold.pkl'
data_path = '/home/jianglb/pythonproject/cloud_segment/data/train_{}_{}'.format(*RESIZE)
# data_path = '/home/noel/pythonproject/cloud_segment/data/train_{}_{}'.format(*RESIZE)

assert str(RESIZE[0]) in data_path and str(RESIZE[1]) in data_path


dir_name = '{}_{}_{}-all'.format(MODEL,ENCODER,time_str)
save_dir = os.path.join(SAVE_PATH, 'segment', dir_name)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
with open(os.path.join(save_dir,'config'), 'w') as f:
    f.write('{}\n\n{}'.format(time_str, utils.config2str(config)))
with open(os.path.join(save_dir,'config.pkl'),'wb') as f:
    pickle.dump(config,f)
print(save_dir)

mean_loss = 0
for fold in range(K):
    if MODEL == 'UNET':
        model = smp.Unet(encoder_name=ENCODER, encoder_weights=encoder_weights,classes=4, activation='sigmoid', ED_drop=ED_drop)
    elif MODEL == "FPN":
        model = smp.FPN(encoder_name=ENCODER, encoder_weights=encoder_weights, classes=4, activation='sigmoid', dropout=dropout, ED_drop=ED_drop)
    else:
        model = None
        raise Exception('Error Model')

    model.cuda()
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=LR,weight_decay=WD)
    criterion = FocalBCEDiceLoss(gamma=gamma)
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, encoder_weights)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=patience)

    if use_apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    if len(device_ids) > 1:
        model = DataParallel(model)
    
    min_loss = 1
    early_stop_counter = 0

    for epoch in range(NUM_EPOCH):
        # train
        dataset = CloudTrainDataset2(train_csv, data_path, kfold_path, fold, phase='train', transform_type=0, preprocessing=utils.get_preprocessing(preprocessing_fn),)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

        for _, imgs, masks, _ in tqdm(dataloader):
            imgs = imgs.float().cuda()
            masks = masks.float().cuda()
            outputs = model(imgs)
            np.save('masks.npy',masks.detach().cpu().numpy(),allow_pickle=True)
            np.save('outputs.npy', outputs.detach().cpu().numpy(),allow_pickle=True)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            if use_apex:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

        # validate
        dataset = CloudTrainDataset2(train_csv, data_path, kfold_path, fold, phase='validate', transform_type=0, preprocessing=utils.get_preprocessing(preprocessing_fn),)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE*2, num_workers=NUM_WORKERS)
        validate_num = len(dataset)
        validate_loss = 0
        with torch.no_grad():
            model.eval()

            for _, imgs, masks, _ in tqdm(dataloader):
                imgs = imgs.float().cuda()
                masks = masks.float().cuda()
                outputs = model(imgs)
                loss = criterion(outputs, masks)

                validate_loss += (loss.item() * imgs.shape[0])

            model.train()
        validate_loss /= validate_num
        lr = optimizer.param_groups[0]['lr']
        print('Fold{} Epoch{}:\tValidate-{:.4f}\tlr-{}e-5'.format(fold, epoch, validate_loss, lr * 100000.))
        scheduler.step(validate_loss)

        if validate_loss < min_loss:
            min_loss = validate_loss
            early_stop_counter = 0
            if len(device_ids) > 1:
                torch.save(model.module.cpu().state_dict(), os.path.join(save_dir, 'model_{}.pth'.format(fold)))
            else:
                torch.save(model.cpu().state_dict(), os.path.join(save_dir, 'model_{}.pth'.format(fold)))
            model.cuda()
        else:
            early_stop_counter += 1
        if early_stop_counter == early_stop:
            mean_loss += min_loss
            break
    print('Fold{} Stop after training {} epoch'.format(fold, epoch - early_stop))
    print('Fold{} Validate Loss:{}'.format(fold, min_loss))
    with open(os.path.join(save_dir, 'config'), 'a') as f:
        f.write('\nFold{} Stop after training {} epoch'.format(fold, epoch - early_stop))
        f.write('\nFold{} Validate Loss:{}\n'.format(fold, min_loss))

with open(os.path.join(save_dir, 'config'), 'a') as f:
    f.write('\nMean Loss:{}\n'.format(mean_loss/K))

print('Done')
print('Mean Loss:', mean_loss/K)
print('Save in:', save_dir)


