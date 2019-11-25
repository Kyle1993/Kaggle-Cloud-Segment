import pandas as pd
import os
import random
import pickle

from global_parameter import *

train_df = pd.read_csv(os.path.join(DATA_PATH,'train.csv'))
train_df['ImageId'] = train_df['Image_Label'].apply(lambda x: x.split('_')[0])
train_df['ClassId'] = train_df['Image_Label'].apply(lambda x: x.split('_')[1])
train_df['hasMask'] = ~ train_df['EncodedPixels'].isna()
train_df = train_df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')

img_ids = train_df.index.tolist()
assert len(img_ids) == len(os.listdir(os.path.join(DATA_PATH,'train_images'))) == (TRAIN_NUM + len(BAD_IMG))

id2name = img_ids
name2id = {}
for i,n in enumerate(id2name):
    name2id[n] = i

# remove bad image & shuffle
bad_id = [name2id[n] for n in BAD_IMG]
shuffle_id = list(set(range(len(img_ids))).difference(set(bad_id)))
assert len(shuffle_id) == TRAIN_NUM
random.shuffle(shuffle_id)

# split
folds = []
for i in range(K):
    low = sum(FOLD_NUM[:i])
    hight = sum(FOLD_NUM[:(i+1)])
    folds.append(shuffle_id[low:hight])

kfold = []
for i in range(K):
    train = list(range(K))
    train.remove(i)

    trainset = []
    for k in train:
        trainset.extend(folds[k])
    validateset = folds[i]

    kfold.append([trainset,validateset])

for i,f in enumerate(folds):
    assert len(f) == FOLD_NUM[i]

all_index = []
for i,s in enumerate(kfold):
    assert len(s[0]) == TRAIN_NUM-FOLD_NUM[i]
    assert len(s[1]) == FOLD_NUM[i]
    assert len(set(s[0]) & set(s[1])) == 0
    all_index.extend(s[1])
assert len(set(all_index)) == TRAIN_NUM
for i in bad_id:
    assert i not in all_index

kfold = {'kfold':kfold, 'id2name':id2name, 'name2id':name2id, 'folds':folds}

with open('kfold.pkl','wb') as f:
    pickle.dump(kfold,f)