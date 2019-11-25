import pandas as pd
import numpy as np
import cv2
import pickle
import os
from tqdm import tqdm
from albumentations import HorizontalFlip, VerticalFlip, Compose, Resize, Normalize
from torchvision import transforms


from global_parameter import *
import utils

RESIZE = (256, 256)
phase = 'train'
save_dir = '{}_{}_{}'.format(phase,RESIZE[0],RESIZE[1]) # 带_的是convertRGB的
img_path = os.path.join(DATA_PATH,'{}_images'.format(phase))
rs = Resize(*RESIZE)

if phase == 'train':
    train_csv_path = os.path.join(DATA_PATH,'train.csv')
    kfold_path = 'kfold.pkl'

    with open(kfold_path, 'rb') as f:
        kfold_info = pickle.load(f)
    folds = kfold_info['folds']
    id2name = kfold_info['id2name']

    train_df = pd.read_csv(train_csv_path)
    train_df['ImageId'] = train_df['Image_Label'].apply(lambda x: x.split('_')[0])
    train_df['ClassId'] = train_df['Image_Label'].apply(lambda x: x.split('_')[1])
    train_df['hasMask'] = ~ train_df['EncodedPixels'].isna()
    train_df = train_df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')

    for i,img_name in enumerate(tqdm(id2name)):
        img = cv2.imread(os.path.join(img_path,img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = train_df.iloc[i][:4]
        mask = utils.make_mask(label,IMAGE_SIZE)

        augmented = rs(image=img, mask=mask)
        # print(augmented['image'].max(),augmented['image'].min())
        # print(augmented['mask'].max(),augmented['mask'].min())
        img = augmented['image'].astype(np.uint8)
        mask = np.around(augmented['mask']*255).astype(np.uint8)

        with open(os.path.join(DATA_PATH,save_dir,'{}.pkl'.format(img_name)),'wb') as f:
            pickle.dump({'image':img, 'mask':mask},f)

elif phase == 'test':
    img_names = os.listdir(img_path)
    for img_name in tqdm(img_names):
        img = cv2.imread(os.path.join(img_path,img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = rs(image=img,)['image']
        img = img.astype(np.uint8)
        # print(img.shape,img.max(),img.min())
        np.save(os.path.join(DATA_PATH,save_dir,'{}.npy'.format(img_name)),img)

else:
    raise Exception('error phase')
