import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import DataParallel

import os
import time
from tqdm import tqdm

from global_parameter import *
from dataset import CloudTrainDataset2
import utils
from loss import FocalBCEDiceLoss
import segmentation_models_pytorch as smp
from apex import amp

import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
device_ids = range(torch.cuda.device_count())
use_apex = True

BATCH_SIZE = len(device_ids) * 8
NUM_WORKERS = len(device_ids) * 3
# data_path = '/home/jianglb/pythonproject/cloud_segment/data/train_480_640'
data_path = '/home/noel/pythonproject/cloud_segment/data/train_480_640'


train_csv = os.path.join(DATA_PATH,'train.csv')
img_path = os.path.join(DATA_PATH,'train_images')
kfold_path = 'kfold.pkl'

time_str = time.strftime("%m%d-%H%M", time.localtime())


config = {'MODEL':'FPN',
          'ENCODER':'densenet201', #efficientnet-b3, resnet34, resnext101_32x8d, resnext101_32x16d,
          'encoder_weights':'imagenet', # instagram
          'NUM_EPOCH':50,
          'RESIZE':(480, 640),
          'LR':5e-4,
          'WD':1e-5,
          'FOLD': 0,
          'early_stop':5,
          'dropout':0.2,
          'ED_drop':0.2,
          'patience':1,
          'gamma':2,
          'note':'',
          'train_time':time_str,
          }
locals().update(config)
assert str(RESIZE[0]) in data_path and str(RESIZE[1]) in data_path

# from hyperboard import Agent
# agent = Agent(username='jlb', password='123',port=5005)
# train_config = config.copy()
# train_config['phase'] = 'train'
# train_loss_record = agent.register(train_config,'loss',overwrite=True)
# validate_config = config.copy()
# validate_config['phase'] = 'validate'
# validate_loss_record = agent.register(validate_config,'loss',overwrite=True)

# dir_name = '{}_{}_{}-fold{}'.format(MODEL,ENCODER,time_str,FOLD)
# save_dir = os.path.join(SAVE_PATH, 'segment', dir_name)
# if not os.path.exists(save_dir):
#     os.mkdir(save_dir)
# with open(os.path.join(save_dir,'config'), 'w') as f:
#     f.write('{}\n\n{}'.format(time_str, utils.config2str(config)))
# print(dir_name)

print(MODEL,ENCODER)

if MODEL == 'UNET':
    model = smp.Unet(encoder_name=ENCODER, encoder_weights=encoder_weights ,classes=4, activation='sigmoid',ED_drop=ED_drop)
elif MODEL == "FPN":
    model = smp.FPN(encoder_name=ENCODER, encoder_weights=encoder_weights, classes=4, activation='sigmoid',dropout=dropout,ED_drop=ED_drop)
else:
    model = None
    raise Exception('Error Model')
model.cuda()
model.train()

optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
criterion = FocalBCEDiceLoss(gamma=gamma)
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER,pretrained=encoder_weights)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=patience)

if use_apex:
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

if len(device_ids) > 1:
    model = DataParallel(model)

min_loss = 1
early_stop_counter = 0
for epoch in range(NUM_EPOCH):
    dataset = CloudTrainDataset2(train_csv, data_path, kfold_path, FOLD, phase='train', transform_type=0,preprocessing=utils.get_preprocessing(preprocessing_fn),)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    train_num = len(dataset)
    train_loss = 0
    for _, imgs, masks, _ in tqdm(dataloader):
        imgs = imgs.float().cuda()
        masks = masks.float().cuda()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        train_loss += (loss.item() * imgs.shape[0])

        optimizer.zero_grad()
        if use_apex:
            with amp.scale_loss(loss, optimizer,) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
    train_loss /= train_num

    # validate
    dataset = CloudTrainDataset2(train_csv, data_path, kfold_path, FOLD, phase='validate', transform_type=0, preprocessing=utils.get_preprocessing(preprocessing_fn),)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE * 2, num_workers=NUM_WORKERS)
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
    scheduler.step(validate_loss)
    print('Epoch{}:\t Train-{:.3f}\tValidate-{:.4f}\tlr-{}e-5'.format(epoch, train_loss, validate_loss, lr*100000.))
    # agent.append(train_loss_record, epoch, train_loss)
    # agent.append(validate_loss_record, epoch, validate_loss)

    if validate_loss < min_loss:
        min_loss = validate_loss
        early_stop_counter = 0
        # if len(device_ids) > 1:
        #     torch.save(model.module.cpu().state_dict(), os.path.join(save_dir, 'model_{}.pth'.format(FOLD)))
        # else:
        #     torch.save(model.cpu().state_dict(), os.path.join(save_dir, 'model_{}.pth'.format(FOLD)))
        # model.cuda()
    else:
        early_stop_counter += 1
    if early_stop_counter == early_stop:
        # print('save in:',save_dir)
        print('Early Stop after training {} epoch'.format(epoch-early_stop))
        print('Validate Loss:',min_loss)
        break


# with open(os.path.join(save_dir,'config'), 'a') as f:
#     f.write('\nEarly Stop after training {} epoch'.format(epoch-early_stop))
#     f.write('\nValidate Loss:{}'.format(min_loss))



