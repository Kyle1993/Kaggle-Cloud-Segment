import numpy as np
import pandas as pd
import cv2
import gc
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Process,Manager
import os

from sklearn.metrics import f1_score,roc_auc_score,fbeta_score

from albumentations import HorizontalFlip, VerticalFlip, ShiftScaleRotate, RandomRotate90, RandomCrop, Compose, Resize, Lambda, GridDistortion, OpticalDistortion, RandomBrightnessContrast, Blur
from albumentations.pytorch import ToTensor, ToTensorV2

import math
import torch
from torch.optim.optimizer import Optimizer, required

import torch


def show_tensor_img(tensor_img):
    img = np.array(tensor_img).transpose((1,2,0))
    fig = plt.figure(figsize=(10,20))
    ax = fig.add_subplot(111)
    ax.imshow(img)
    plt.show()


def show_tensor_img_mask(tensor_img, tensor_mask):
    img = np.array(tensor_img).transpose((1,2,0))
    mask = np.array(tensor_mask)
    mask[0] = mask[0] * 1
    mask[1] = mask[1] * 5
    mask[2] = mask[2] * 10
    mask[3] = mask[3] * 50

    mask =mask.transpose((1,2,0)).sum(axis=2)

    fig = plt.figure(figsize=(15, 50))
    ax1, ax2 = fig.subplots(1, 2, )

    ax1.imshow(img)
    ax2.imshow(mask)
    plt.show()


def to_tensor(x, **kwargs):
    """
    Convert image or mask.
    """
    return x.transpose(2, 0, 1).astype('float32')


def get_transforms(phase, transform_type=None, resize=None):
    list_transforms = []

    if resize is not None:
        list_transforms.append(Resize(*resize))

    if phase == "train":
        list_transforms.extend([
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                Blur(blur_limit=1, p=0.5),
                ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5, border_mode=0),
                # OpticalDistortion(0.05, 0.05, p=0.5),
                # RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1,brightness_by_max=False, p=0.5),
                # ShiftScaleRotate(scale_limit=0.1, rotate_limit=0, shift_limit=0.1, p=0.2, border_mode=0),
                GridDistortion(p=0.5,),
                # OpticalDistortion(p=0.5, distort_limit=1, shift_limit=0.1),
        ])


    elif phase == 'validate':
        assert transform_type is not None
        if transform_type == 0:
            pass
        elif transform_type == 1:
            list_transforms.append(HorizontalFlip(p=1))
        elif transform_type == 2:
            list_transforms.append(VerticalFlip(p=1))
        elif transform_type == 3:
            list_transforms.extend([HorizontalFlip(p=1),
                                    VerticalFlip(p=1),])
        else:
            raise Exception('Error transform type!')

    elif phase == 'test':
        assert transform_type is not None
        if transform_type == 0:
            pass
        elif transform_type == 1:
            list_transforms.append(HorizontalFlip(p=1))
        elif transform_type == 2:
            list_transforms.append(VerticalFlip(p=1))
        elif transform_type == 3:
            list_transforms.extend([HorizontalFlip(p=1),
                                    VerticalFlip(p=1),])
        else:
            raise Exception('Error transform type!')

    else:
        raise Exception('Error phase!')
    # list_transforms.append(ToTensorV2())
    # list_transforms.extend([
    #     Lambda(image=preprocessing_fn),
    #     Lambda(image=to_tensor, mask=to_tensor),
    # ])

    transformer = Compose(list_transforms)

    return transformer


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        Lambda(image=preprocessing_fn),
        Lambda(image=to_tensor, mask=to_tensor),
    ]
    return Compose(_transform)

def area(encode_pixel):
    if pd.isna(encode_pixel):
        return 0
    ep = encode_pixel.split(' ')
    assert len(ep) % 2 == 0
    sum_area = 0
    for n in ep[1::2]:
        sum_area += int(n)
    return sum_area

def rle2mask(rle_string, img_size, dtype=np.float):
    '''
    convert RLE(run length encoding) string to numpy array
x
    Parameters:
    rle_string (str): string of rle encoded mask
    height (int): height of the mask
    width (int): width of the mask

    Returns:
    numpy.array: numpy array of the mask
    '''

    rows, cols = img_size

    if isinstance(rle_string, str):
        rle_numbers = [int(num_string) for num_string in rle_string.split(' ')]
        rle_pairs = np.array(rle_numbers).reshape(-1, 2)
        img = np.zeros(rows * cols, dtype=dtype)
        for index, length in rle_pairs:
            index -= 1
            img[index:index + length] = 1
        img = img.reshape(cols, rows)
        img = img.T
        return img
    else:
        return np.zeros(img_size,dtype=dtype)

def mask2rle(mask):
    '''
    Convert mask to rle.
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def remove_small(mask, cpn_threshod, save1=False):
    """
    Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored
    """
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros(mask.shape, np.float32)
    max_area = -1
    max_id = -1
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > max_area:
            max_id = c
            max_area = p.sum()
        if p.sum() > cpn_threshod:
            predictions[p] = 1

    if save1 and predictions.sum() == 0:
        p = (component == max_id)
        predictions[p] = 1

    return predictions

def to_convex(mask):
    reshape_mask = np.zeros(mask.shape)
    contours, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        hull = cv2.convexHull(c)
        cv2.drawContours(reshape_mask, [hull], 0, (255, 255, 255), -1)
    reshape_mask = np.around(reshape_mask / 255.).astype(np.uint8)
    return reshape_mask

def to_polygon(mask):
    reshape_mask = np.zeros(mask.shape)
    contours, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        epsilon = 0.02 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        cv2.drawContours(reshape_mask, [approx], 0, (255, 255, 255), -1)
    reshape_mask = np.around(reshape_mask / 255.).astype(np.uint8)
    return reshape_mask

class Mask2Rle(object):
    def __init__(self, label_names, resize=None, seg_threshold=0.8, cpn_threshold=0,):
        if isinstance(seg_threshold, int) or isinstance(seg_threshold, float):
            seg_threshold = [seg_threshold] * 4
        if isinstance(cpn_threshold, int) or isinstance(cpn_threshold, float):
            cpn_threshold = [cpn_threshold] * 4

        self.seg_threshold = seg_threshold
        self.cpn_threshold = cpn_threshold
        self.label_names = label_names

        self.resizer = None

        if resize is not None:
            self.resizer = Resize(*resize,)

    def __call__(self, img_name, mask, save_i):
        '''
        mask range [0,100]
        '''
        if isinstance(mask, torch.Tensor):
            mask = np.array(mask)

        mask = mask.transpose((1,2,0))

        if self.resizer is not None:
            mask = self.resizer(image=mask)['image']
        mask = mask.transpose((2,0,1))
        mask_confidence = mask.copy()

        for i in range(len(self.label_names)):
            mask[i][mask[i] <= (self.seg_threshold[i]*100)] = 0
            mask[i][mask[i] >(self.seg_threshold[i]*100)] = 1

            mask_confidence[i][mask_confidence[i] <= 95] = 0
            mask_confidence[i][mask_confidence[i] > 95] = 1

        Image_Label = []
        EncodedPixels = []
        cm = []
        for i,l in enumerate(self.label_names):
            Image_Label.append('{}_{}'.format(img_name,l))
            # # try at least 1 positive pred
            # if i == save_i:
            #     mask_ = remove_small(mask[i],self.cpn_threshold[i],save1=True)
            # else:
            #     mask_ = remove_small(mask[i],self.cpn_threshold[i])
            mask_ = remove_small(mask[i], self.cpn_threshold[i])
            mask_ = to_convex(mask_)
            mask_ = np.logical_or(mask_,mask_confidence[i])
            cm.append(mask_)

            EncodedPixels.append(mask2rle(mask_))

        cm = np.concatenate(cm,axis=0)

        return Image_Label, EncodedPixels, cm

    def dice(self,tg_mask,pred_mask,):
        assert tg_mask.shape == pred_mask.shape
        # tg_mask = tg_mask.reshape((-1,*tg_mask.shape[-2:]))
        # pred_mask = pred_mask.reshape((-1,*pred_mask.shape[-2:]))
        d = []
        for i in tqdm(range(tg_mask.shape[0])):
            for j in range(4):
                t = tg_mask[i,j]
                p = pred_mask[i,j]

                t[t <= 50] = 0
                t[t > 50] = 1

                p[p <= (self.seg_threshold[j] * 100)] = 0
                p[p > (self.seg_threshold[j] * 100)] = 1

                p = remove_small(p, self.cpn_threshold[j])
                p = to_convex(p)

                d.append(dice(t,p))
        return np.mean(d)


    def batch_convert(self, names, masks, max_prob):
        if isinstance(masks, torch.Tensor):
            masks = np.array(masks)

        Image_Label_batch = []
        EncodedPixels_batch = []
        for i in tqdm(range(len(names))):
            names_, rles_, _ = self.__call__(names[i],masks[i],max_prob[i])
            Image_Label_batch.extend(names_)
            EncodedPixels_batch.extend(rles_)
        return Image_Label_batch, EncodedPixels_batch

def make_mask(label, img_size, dtype=np.float):
    mask = []
    for i in range(label.shape[0]):
        mask.append(rle2mask(label[i], img_size, dtype=dtype))
    mask = np.asarray(mask)
    mask = mask.transpose((1,2,0))
    return mask

# def dict_merge(dl):
#     assert len(dl) > 1
#     res = dl[0].copy()
#     for d in dl[1:]:
#         res.update(d)
#     return res

def dice_float(img1,img2):
    img1 = np.asarray(img1).astype(np.float)
    img2 = np.asarray(img2).astype(np.float)
    if img1.sum() == 0 and img2.sum() == 0:
        return 1
    else:
        return 2. * (img1 * img2).sum() / (img1.sum() + img2.sum())


def dice(img1, img2):
    img1 = np.asarray(img1).astype(np.bool)
    img2 = np.asarray(img2).astype(np.bool)

    if img1.sum() == 0 and img2.sum() == 0:
        return 1

    intersection = np.logical_and(img1, img2)
    return 2. * intersection.sum() / (img1.sum() + img2.sum())


def soft_dice_cuda(input, target,):
    assert input.shape == target.shape

    input = input.view(-1, *input.shape[-2:])
    target = target.view(-1, *target.shape[-2:])
    eps = 1e-7

    intersection = torch.sum((input * target),dim=(1,2))
    score = (2. * intersection + eps) / (torch.sum(input,dim=(1,2)) + torch.sum(target,dim=(1,2)) + eps)

    return score.mean().item()

def soft_dice(input, target,):
    '''
    input & target range [0,100]
    '''
    assert input.shape == target.shape

    input = input.reshape((-1, *input.shape[-2:]))
    target = target.reshape((-1, *target.shape[-2:]))
    eps = 1e-7

    intersection = (input * target).sum(1).sum(1) / 100
    score = (2. * intersection + eps) / (input.sum(1).sum(1) + target.sum(1).sum(1) + eps)

    return score.mean()

def search_threshold(masks, seg_probs, classes, cls_probs,):
    '''
    masks range [0,100]
    seg_probs range [0,100]
    '''
    cls_threshold = []
    for class_id in range(4):
        best_class_threshold = 0
        max_score = 0
        for ct in range(0,100,10):
            ct /= 100
            c = cls_probs[:,class_id].copy()
            c[c>ct] = 1
            c[c<=ct] = 0
            score = f1_score(classes[:,class_id],c)
            # score = roc_auc_score()
            # score = fbeta_score(classes[:,class_id],c,beta=1)
            if score>max_score:
                max_score = score
                best_class_threshold = ct
        cls_threshold.append(best_class_threshold)
        print(class_id, best_class_threshold, max_score)

    cls_pred = np.zeros_like(cls_probs).astype(np.uint8)
    for class_id in range(4):
        c = cls_probs[:, class_id].copy()
        c[c > cls_threshold[class_id]] = 1
        c[c <= cls_threshold[class_id]] = 0
        cls_pred[:,class_id] = c.astype(np.uint8)
    cls_pred = np.tile(cls_pred.reshape(-1,1,1),(1,seg_probs.shape[-2],seg_probs.shape[-1]))
    seg_probs = seg_probs.reshape((-1,*seg_probs.shape[-2:]))
    masks = masks.reshape((-1,*masks.shape[-2:]))

    assert len(cls_pred) == len(seg_probs)

    seg_probs *= cls_pred

    mask = np.zeros((masks.shape[0],350,525),dtype=np.uint8)
    prob = np.zeros((masks.shape[0],350,525),dtype=np.uint8)

    for i in range(masks.shape[0]):
        m = cv2.resize(masks[i], dsize=(525,350)) # (525,350)
        p = cv2.resize(seg_probs[i], dsize=(525,350))
        mask[i] = np.around(m/100.).astype(np.uint8)
        prob[i] = np.around(p).astype(np.uint8)

    del masks
    del seg_probs
    gc.collect()

    # seg_params = []
    seg_threshold = []
    cpn_threshold = []
    best_dice = []
    for class_id in range(4):
        # print(class_id)
        attempts = []
        for st in tqdm(range(0, 100, 10)):
            for ct in range(5000,30001,5000):
                d = []
                for i in range(class_id, len(prob), 4):
                    p = prob[i].copy()
                    p_confidence = prob[i].copy()
                    m = mask[i].copy()

                    p[p<=st] = 0
                    p[p>st] = 1
                    p_confidence[p_confidence <= 95] = 0
                    p_confidence[p_confidence > 95] = 1

                    p = remove_small(p,ct)
                    p = to_convex(p)
                    p = np.logical_or(p,p_confidence)

                    d.append(dice(p,m))

                attempts.append((st/100.,ct,np.mean(d)))
        # break
        attempts_df = pd.DataFrame(attempts, columns=['seg_threshold', 'size', 'dice'])
        attempts_df = attempts_df.sort_values('dice', ascending=False)
        # print(attempts_df.head(5))

        seg_threshold.append(attempts_df['seg_threshold'].values[0])
        cpn_threshold.append(attempts_df['size'].values[0])
        best_dice.append(attempts_df['dice'].values[0])

        # seg_params.append((cls_threshold[class_id], best_seg_threshold, best_size, best_dice))

    print('\n\n\n \tcls_threshold\tseg_threshold\tcpn_threshold\tbest dice')
    scores = 0
    for i in range(4):
        print('{}\t{}\t{}\t{}\t{}'.format(i, cls_threshold[i],seg_threshold[i],cpn_threshold[i],best_dice[i]))
        scores += best_dice[i]
    scores /= 4.

    print('Final Score:',scores)

    return cls_threshold, seg_threshold, cpn_threshold, best_dice

def config2str(config):
    s = ''
    for i,v in sorted(config.items(), key=lambda x:x[0]):
        s += '{}:\t{}\n'.format(i,v)
    return s

if __name__ == '__main__':


    p = np.array([[0, 1, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 1],],dtype=np.float)

    mask = remove_small(p,2,save1=False)
    print(mask)






