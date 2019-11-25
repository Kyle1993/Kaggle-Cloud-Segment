import torch
from torch import optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

import os
import time

import utils
from global_parameter import *
from dataset import CloudTrainDataset,CloudTrainDataset2

from hyperboard import Agent

BATCH_SIZE = 16
FOLD = 0
GPU = 1
VALIDATE_STEP = 10
NUM_WORKERS = 4
data_path = '/home/jianglb/pythonproject/cloud_segment/data/train_352_544_'

'''
GPU  RESIZE      note


'''
time_str = time.strftime("%m%d-%H%M", time.localtime())
config = {'ENCODER':'resnet34',
          'NUM_EPOCH':50,
          'RESIZE':(352, 544),
          'LR':5e-4,
          'note':'convertRGB+ReduceLROnPlateau-0.5-20',# ReduceLROnPlateau-0.5-20, MultiStepLR-5,20-0.025, MultiStepLR-10,14,16,18,20,22,24-0.2
          'train_time':time_str,
          }
locals().update(config)

assert str(RESIZE[0]) in data_path and str(RESIZE[1]) in data_path

dir_name = '{}_{}'.format(ENCODER,time_str)
save_dir = os.path.join(SAVE_PATH, 'segment', dir_name)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
with open(os.path.join(save_dir,'config'), 'w') as f:
    f.write('{}\n\n{}'.format(time_str, utils.config2str(config)))


agent = Agent(username='jlb', password='123',port=5005)
train_config = config.copy()
train_config['phase'] = 'train'
train_loss_record = agent.register(train_config,'loss',overwrite=True)
validate_config = config.copy()
validate_config['phase'] = 'validate'
validate_loss_record = agent.register(validate_config,'loss',overwrite=True)

lr_config = config.copy()
lr_config['phase'] = 'learning rate'
lr_record = agent.register(lr_config,'lr',overwrite=True)

train_csv = os.path.join(DATA_PATH,'train.csv')
img_path = os.path.join(DATA_PATH,'train_images')
kfold_path = 'kfold.pkl'

model = smp.Unet(encoder_name=ENCODER, classes=4, activation='sigmoid')
model.cuda(GPU)
model.train()
optimizer = optim.Adam(model.parameters(), lr=LR)
# optimizer = utils.RAdam(model.parameters(), lr=LR)
# optimizer = torch.optim.Adam([
#     {'params': model.decoder.parameters(), 'lr': 1e-2},
#     {'params': model.encoder.parameters(), 'lr': 1e-3},
# ])
# criterion = torch.nn.BCEWithLogitsLoss()
criterion = smp.utils.losses.BCEDiceLoss(eps=1.)
# 记住修改 scheduler.step 的位置
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=50)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.025,)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,14,16,18,20], gamma=0.2)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, 'imagenet')

# validate_dataset = CloudTrainDataset(train_csv, img_path, kfold_path, FOLD, phase='validate', transform_type=0 , img_size=IMAGE_SIZE, resize=RESIZE, mean=MEAN, std=STD)
validate_dataset = CloudTrainDataset2(train_csv, data_path, kfold_path, FOLD, phase='validate', transform_type=0, preprocessing=utils.get_preprocessing(preprocessing_fn), mean=MEAN, std=STD,)


global_step = 0
for epoch in range(NUM_EPOCH):
    # dataset = CloudTrainDataset(train_csv, img_path, kfold_path, fold=FOLD, phase='train', transform_type=None, img_size=IMAGE_SIZE, resize=RESIZE, mean=MEAN, std=STD)
    dataset = CloudTrainDataset2(train_csv, data_path, kfold_path, FOLD, phase='train', transform_type=0, preprocessing=utils.get_preprocessing(preprocessing_fn), mean=MEAN, std=STD, )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    for i,(img_ids, imgs, masks, _) in enumerate(dataloader):
        imgs = imgs.float().cuda(GPU)
        masks = masks.float().cuda(GPU)
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        if epoch>0: # skip unstable phase
            agent.append(train_loss_record, global_step, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        global_step += 1

        if global_step % VALIDATE_STEP == 0:
            train_loss = loss.item()
            model.eval()
            with torch.no_grad():
                _, imgs, masks, _ = validate_dataset.sample(BATCH_SIZE * 2)
                imgs = imgs.float().cuda(GPU)
                masks = masks.float().cuda(GPU)
                outputs = model(imgs)
                loss = criterion(outputs, masks)

                lr = optimizer.param_groups[0]['lr']
                print('Epoch{}[{}/{} {:.1f}%]:\t Train-{:.4f}\tValidate-{:.4f}\tlr-{}e-5'.format(epoch, i+1, len(dataloader), i*BATCH_SIZE/len(dataset)*100., train_loss, loss.item(), lr*100000.))

                if epoch>0:
                    agent.append(validate_loss_record, global_step, loss.item())
                agent.append(lr_record, global_step, lr)

            model.train()
            scheduler.step(loss.item())

    # scheduler.step()

    if (epoch+1) % 3 == 0:
        torch.save(model.cpu().state_dict(), os.path.join(save_dir,'model_{}.pth'.format(epoch+1)))
        model.cuda(GPU)







