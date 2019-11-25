import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import segmentation_models_pytorch as smp
from apex import amp

import os
import time
from tqdm import tqdm

import utils
from global_parameter import *
from dataset import CloudTrainDataset2
from models import load_model

import warnings

warnings.filterwarnings('ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
device_ids = range(torch.cuda.device_count())

time_str = time.strftime("%m%d-%H%M", time.localtime())
config = {'CLASSIFER':'resnext50_32x4d',
          'encoder_weights':'imagenet',
          'NUM_EPOCH':40,
          'RESIZE':(480, 640),
          'LR':5e-4,
          'WD':1e-5,
          'dropout':0.2,
          'patience':3,
          'early_stop': 6,
          'train_time':time_str,
          }

locals().update(config)

fold = 0
NUM_WORKERS = len(device_ids) * 4
BATCH_SIZE = len(device_ids) * 16
# data_path = '/home/jianglb/pythonproject/cloud_segment/data/train_{}_{}'.format(*RESIZE)
data_path = '/home/noel/pythonproject/cloud_segment/data/train_{}_{}'.format(*RESIZE)

train_csv = os.path.join(DATA_PATH,'train.csv')
kfold_path = 'kfold.pkl'

# dir_name = '{}_{}_all'.format(CLASSIFER, time_str)
# save_dir = os.path.join(SAVE_PATH, 'classify', dir_name)
# if not os.path.exists(save_dir):
#     os.mkdir(save_dir)
# with open(os.path.join(save_dir, 'config'), 'w') as f:
#     f.write('{}\n\n{}'.format(time_str, utils.config2str(config)))

model = load_model(CLASSIFER,classes=4,dropout=dropout,pretrained=True)
model.cuda()
model.train()

optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
criterion = torch.nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=patience)
preprocessing_fn = smp.encoders.get_preprocessing_fn('resnet34', 'imagenet')

model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

if len(device_ids) > 1:
    model = DataParallel(model)

min_loss = 1
early_stop_counter = 0

for epoch in range(NUM_EPOCH):
    dataset = CloudTrainDataset2(train_csv, data_path, kfold_path, fold, phase='train', transform_type=0, preprocessing=utils.get_preprocessing(preprocessing_fn),)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,shuffle=True)
    for _, imgs, _, classes in tqdm(dataloader):
        imgs = imgs.float().cuda()
        classes = classes.float().cuda()
        outputs = model(imgs)
        loss = criterion(outputs, classes)

        optimizer.zero_grad()
        # loss.backward()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

    dataset = CloudTrainDataset2(train_csv, data_path, kfold_path, fold, phase='validate', transform_type=0, preprocessing=utils.get_preprocessing(preprocessing_fn),)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE*2, num_workers=NUM_WORKERS, shuffle=True)
    validate_loss = 0
    validate_num = len(dataset)
    with torch.no_grad():
        model.eval()
        for _, imgs, _, classes in tqdm(dataloader):
            imgs = imgs.float().cuda()
            classes = classes.float().cuda()
            outputs = model(imgs)
            loss = criterion(outputs, classes)

            validate_loss += (loss.item() * imgs.shape[0])
        model.train()
    validate_loss /= validate_num
    scheduler.step(validate_loss)
    lr = optimizer.param_groups[0]['lr']
    print('Fofl{} Epoch{}:\tValidate-{:.4f}\tlr-{}e-5'.format(fold, epoch, validate_loss, lr * 100000.))

    if validate_loss < min_loss:
        min_loss = validate_loss
        early_stop_counter = 0
        # torch.save(model.module.cpu().state_dict(), os.path.join(save_dir, 'model_{}.pth'.format(fold)))
        # # torch.save(model.cpu().state_dict(), os.path.join(save_dir, 'model_{}.pth'.format(fold)))
        # model.cuda()
    else:
        early_stop_counter += 1
    if early_stop_counter == early_stop:
        print('Fold{} Stop after training {} epoch'.format(fold,epoch-early_stop))
        print('Fold{} Validate Loss:{}'.format(fold,min_loss))
        # with open(os.path.join(save_dir, 'config'), 'a') as f:
        #     f.write('\nFold{} Stop after training {} epoch'.format(fold,epoch-early_stop))
        #     f.write('\nFold{} Validate Loss:{}\n'.format(fold,min_loss))
        break







