import torch
from torch import optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

import os
import time
from tqdm import tqdm
import pickle

import utils
from global_parameter import *
from dataset import CloudTrainDataset,CloudTrainDataset2
from loss import MixedLoss

from hyperboard import Agent

import warnings
warnings.filterwarnings('ignore')

BATCH_SIZE = 16
GPU = 2
VALIDATE_STEP = 10
NUM_WORKERS = 3
data_path = '/home/jianglb/pythonproject/cloud_segment/data/train_480_640'

'''
GPU  RESIZE      note


'''
time_str = time.strftime("%m%d-%H%M", time.localtime())
config = {'MODEL':'FPN',
          'ENCODER':'resnet34', #efficientnet-b3,resnet34
          'NUM_EPOCH':40,
          'RESIZE':(480, 640),
          'LR':5e-4,
          'WD':1e-5,
          'FOLD': 1,
          'dropout':0.2,
          'ED_drop':0.4,
          'note':'fpn-resnet34-fold0',# ReduceLROnPlateau-0.5-20, MultiStepLR-5,20-0.025, MultiStepLR-10,14,16,18,20,22,24-0.2
          'train_time':time_str,
          }
locals().update(config)

assert str(RESIZE[0]) in data_path and str(RESIZE[1]) in data_path

# dir_name = '{}_{}_{}-S'.format(MODEL,ENCODER,time_str)
# save_dir = os.path.join(SAVE_PATH, 'segment', dir_name)

# agent = Agent(username='jlb', password='123',port=5007)
# train_config = config.copy()
# train_config['phase'] = 'train'
# train_loss_record = agent.register(train_config,'loss',overwrite=True)
#
# validate_config = config.copy()
# validate_config['phase'] = 'validate'
# validate_loss_record = agent.register(validate_config,'loss',overwrite=True)
#
# lr_config = config.copy()
# lr_config['phase'] = 'learning rate'
# lr_record = agent.register(lr_config,'lr',overwrite=True)

train_csv = os.path.join(DATA_PATH,'train.csv')
img_path = os.path.join(DATA_PATH,'train_images')
kfold_path = 'kfold.pkl'

record_dir = 'record'
record_name = '{}_{}_{}'.format(MODEL,ENCODER,time_str)
record_note = utils.config2str(config)

if MODEL == 'UNET':
    model = smp.Unet(encoder_name=ENCODER, classes=4, activation='sigmoid',ED_drop=ED_drop)
elif MODEL == "FPN":
    model = smp.FPN(encoder_name=ENCODER, classes=4, activation='sigmoid',dropout=dropout,ED_drop=ED_drop)
else:
    model = None
    raise Exception('Error Model')
model.cuda(GPU)
model.train()

optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
# optimizer = utils.RAdam(model.parameters(), lr=LR)
# optimizer = torch.optim.Adam([
#     {'params': model.decoder.parameters(), 'lr': 1e-2},
#     {'params': model.encoder.parameters(), 'lr': 1e-3},
# ])

criterion = smp.utils.losses.BCEDiceLoss(eps=1.)
# criterion = torch.nn.BCEWithLogitsLoss()
# criterion = MixedLoss(alpha=1,gamma=2)

# 记住修改 scheduler.step 的位置
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=1)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.025,)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,14,16,18,20], gamma=0.2)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, 'imagenet')

train_r = []
validate_r = []
lr_r = []
# train_loss = 0
# validate_loss = 10
# train_dice = 0
# validate_dice = 0
for epoch in range(NUM_EPOCH):
    # train
    dataset = CloudTrainDataset2(train_csv, data_path, kfold_path, FOLD, phase='train', transform_type=0, preprocessing=utils.get_preprocessing(preprocessing_fn), mean=MEAN, std=STD, )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    train_num = len(dataset)
    train_loss = 0
    train_dice = 0
    for img_ids, imgs, masks,_ in tqdm(dataloader):
        imgs = imgs.float().cuda(GPU)
        masks = masks.float().cuda(GPU)
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        train_loss += (loss.item() * imgs.shape[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        d = utils.batch_dice(torch.sigmoid(outputs).detach().cpu(), masks.detach().cpu())
        train_dice += (d * imgs.shape[0])

    train_loss /= train_num
    train_dice /= train_num

    # validate
    dataset = CloudTrainDataset2(train_csv, data_path, kfold_path, FOLD, phase='validate', transform_type=0, preprocessing=utils.get_preprocessing(preprocessing_fn), mean=MEAN, std=STD, )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE*2, num_workers=NUM_WORKERS)
    validate_num = len(dataset)
    validate_loss = 0
    validate_dice = 0
    with torch.no_grad():
        model.eval()

        for _, imgs, masks, _ in tqdm(dataloader):
            imgs = imgs.float().cuda(GPU)
            masks = masks.float().cuda(GPU)
            outputs = model(imgs)
            loss = criterion(outputs, masks)

            validate_loss += (loss.item() * imgs.shape[0])
            d = utils.batch_dice(torch.sigmoid(outputs).cpu(), masks.cpu())
            validate_dice += (d * imgs.shape[0])

        model.train()
    validate_loss /= validate_num
    validate_dice /= validate_num
    # if epoch % 5 == 0:
    #     validate_loss /= 2
    lr = optimizer.param_groups[0]['lr']

    print('Epoch{}:\t Train-{:.3f}({:.3f})\tValidate-{:.4f}({:.3f})\tlr-{}e-5'.format(epoch, train_loss, train_dice, validate_loss, validate_dice, lr*100000.))

    scheduler.step(validate_loss)

    # agent.append(train_loss_record, epoch, train_loss)
    # agent.append(validate_loss_record, epoch, validate_loss)
    # agent.append(lr_record, epoch, lr)

    train_r.append((epoch,train_loss))
    validate_r.append((epoch,validate_loss))
    lr_r.append((epoch,lr))

    record = {'train':{'record':train_r,'name':'train_{}'.format(record_name),'note':record_note},
              'validate':{'record':validate_r,'name':'validate_{}'.format(record_name),'note':record_note},
              'lr':{'record':lr_r,'name':'lr_{}'.format(record_name),'note':record_note}}

    with open(os.path.join(record_dir,'{}.pkl'.format(record_name)),'wb') as f:
        pickle.dump(record,f)







