import pandas as pd
import numpy as np
import os
import cv2
import pickle
import random

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import utils

class CloudTrainDataset(Dataset):
    def __init__(self, csv_path, img_path, kfold_path, fold, phase, transform_type, img_size, resize, mean, std):
        super(CloudTrainDataset,self).__init__()

        assert phase == 'train' or phase == 'validate'
        self.img_path = img_path
        self.phase = phase
        self.img_size = img_size

        with open(kfold_path, 'rb') as f:
            kfold_info = pickle.load(f)
        self.indexs = kfold_info['kfold'][fold][0] if phase=='train' else kfold_info['kfold'][fold][1]
        self.id2name = kfold_info['id2name']

        self.train_df = pd.read_csv(csv_path)
        self.train_df['ImageId'] = self.train_df['Image_Label'].apply(lambda x: x.split('_')[0])
        self.train_df['ClassId'] = self.train_df['Image_Label'].apply(lambda x: x.split('_')[1])
        self.train_df['hasMask'] = ~ self.train_df['EncodedPixels'].isna()
        self.train_df = self.train_df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')

        self.transformer = utils.get_transforms(phase=phase, transform_type=transform_type, resize=resize)
        self.normalize = transforms.Normalize(mean,std)

    def __getitem__(self, i):
        index = self.indexs[i]
        img_name = self.id2name[index]
        label = self.train_df.iloc[index][:4]
        cls = torch.from_numpy(np.array(~pd.isna(label),dtype=np.int))
        mask = utils.make_mask(label,self.img_size)
        img = cv2.imread(os.path.join(self.img_path,img_name)).astype(np.float)

        augmented = self.transformer(image=img, mask=mask)
        img = augmented['image'] / 255
        img = self.normalize(img)
        mask = augmented['mask']
        mask = mask.permute(2,0,1)

        return img_name, img, mask, cls

    def sample(self, batch_size):
        batch_indexs = random.sample(list(range(len(self.indexs))),batch_size)
        batch = []
        for index in batch_indexs:
            batch.append(self.__getitem__(index))

        img_names, imgs, masks, classes = zip(*batch)
        imgs = torch.stack(imgs)
        masks = torch.stack(masks)
        classes = torch.stack(classes)

        return img_names, imgs, masks, classes

    def __len__(self):
        return len(self.indexs)

class CloudTrainDataset2(Dataset):
    def __init__(self, csv_path, data_path, kfold_path, fold, phase, transform_type, preprocessing,):
        super(CloudTrainDataset2,self).__init__()

        assert phase == 'train' or phase == 'validate'
        self.data_path = data_path
        self.phase = phase

        with open(kfold_path, 'rb') as f:
            kfold_info = pickle.load(f)
        self.indexs = kfold_info['kfold'][fold][0] if phase=='train' else kfold_info['kfold'][fold][1]
        self.id2name = kfold_info['id2name']

        self.train_df = pd.read_csv(csv_path)
        self.train_df['ImageId'] = self.train_df['Image_Label'].apply(lambda x: x.split('_')[0])
        self.train_df['ClassId'] = self.train_df['Image_Label'].apply(lambda x: x.split('_')[1])
        self.train_df['hasMask'] = ~ self.train_df['EncodedPixels'].isna()
        self.train_df = self.train_df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')

        self.transformer = utils.get_transforms(phase=phase, transform_type=transform_type, resize=None)
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        index = self.indexs[i]
        img_name = self.id2name[index]

        with open(os.path.join(self.data_path,'{}.pkl'.format(img_name)),'rb') as f:
            try:
                data = pickle.load(f)
            except:
                print(os.path.join(self.data_path, '{}.pkl'.format(img_name)))
        img = data['image'].astype(np.float)
        mask = data['mask'].astype(np.float)

        label = self.train_df.iloc[index][:4]
        cls = np.array(~pd.isna(label),dtype=np.int32)


        augmented = self.transformer(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask'] / 255

        if self.preprocessing:
            preprocessed = self.preprocessing(image=img, mask=mask)
            img = preprocessed['image']
            mask = preprocessed['mask']

        # img = augmented['image'] / 255
        # img = self.normalize(img)
        # mask = augmented['mask'] / 255
        # mask = mask.permute(2,0,1)

        return img_name, img, mask, cls

    def sample(self, batch_size):
        batch_indexs = random.sample(list(range(len(self.indexs))),batch_size)
        batch = []
        for index in batch_indexs:
            batch.append(self.__getitem__(index))

        img_names, imgs, masks, classes = zip(*batch)

        # imgs = torch.stack(imgs)
        # masks = torch.stack(masks)
        # classes = torch.stack(classes)

        imgs = torch.from_numpy(np.asarray(imgs))
        masks = torch.from_numpy(np.asarray(masks))
        classes = torch.from_numpy(np.asarray(classes))

        return img_names, imgs, masks, classes

    def __len__(self):
        return len(self.indexs)

class CloudTestDataset(Dataset):
    def __init__(self, data_path, transform_type, preprocessing,):
        super(CloudTestDataset, self).__init__()
        self.data_path = data_path
        self.img_names = os.listdir(data_path)

        self.preprocessing = preprocessing
        self.transform_type = transform_type
        self.transformer = utils.get_transforms(phase='test',transform_type=transform_type,resize=None)

    def __getitem__(self, i):
        img_name = self.img_names[i]
        img = np.load(os.path.join(self.data_path,img_name)).astype(np.float)
        img = self.transformer(image=img)['image']
        if self.preprocessing:
            preprocessed = self.preprocessing(image=img)
            img = preprocessed['image']

        return img_name[:-4], img, self.transform_type

    def __len__(self):
        return len(self.img_names)

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from global_parameter import *
    import segmentation_models_pytorch as smp

    # DATA_PATH = '/media/kyle/Data/Data/cloud_segment'

    with open('kfold.pkl','rb') as f:
        kfold = pickle.load(f)

    train_csv = os.path.join(DATA_PATH, 'train.csv')
    img_path = os.path.join(DATA_PATH, 'train_images')
    kfold_path = 'kfold.pkl'

    FOLD = 0
    data_path = '/home/jianglb/pythonproject/cloud_segment/data/train_480_640'
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, 'imagenet')

    dataset = CloudTrainDataset2(train_csv, data_path, kfold_path, FOLD, phase='validate', transform_type=0,
                                          preprocessing=utils.get_preprocessing(preprocessing_fn), mean=MEAN, std=STD, )

    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    # print(len(dataloader))
    # name, img, mask, cls = dataset.sample(24)
    # print(len(name),img.shape, mask.shape, cls.shape)
    # print(img.dtype, mask.dtype, cls.dtype)

    for name, img, mask, _ in dataloader:
        print(len(name), img.shape, mask.shape)
        print(img.min(), img.max())
        print(mask.min(), mask.max())

