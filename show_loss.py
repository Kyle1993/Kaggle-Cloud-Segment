import os
from global_parameter import *
import pickle
'''
Ensemble13_1118-0443
Ensemble13_max_1117-0243
Ensemble14_1114-1411
Ensemble14_1114-1720
Ensemble17_1116-1549
Ensemble17_max_1116-1746
Ensemble3_1118-0414
Ensemble3_max_1118-0239




FPN_dpn68b_1118-1327-all

'''
folders = ['FPN_resnext101_32x8d_1107-2310-all',
           'FPN_efficientnet-b3_1109-0848-all',
           'FPN_efficientnet-b3_1118-0057-all',
           'FPN_efficientnet-b3_1104-1734-all',
           'FPN_efficientnet-b0_1113-2136-all',
           'FPN_resnet34_1110-1049-all',
           'FPN_resnet34_1113-1740-all',
           'FPN_resnet50_1110-1640-all',
           'FPN_resnet101_1110-1041-all',
           'FPN_densenet161_1110-2050-all',
           'FPN_densenet169_1110-2222-all',
           'FPN_densenet201_1112-1425-all',
           'FPN_dpn92_1111-2051-all',
           'FPN_dpn68b_1111-2353-all',
           'FPN_se_resnet50_1114-1138-all',
           'UNET_efficientnet-b3_1105-0551-all',
           'UNET_resnet34_1111-0558-all',
           'UNET_resnext101_32x8d_1113-0102-all',
           'UNET_densenet161_1115-2032-all',
           'UNET_dpn92_1116-1102-all',

           'FPN_resnext101_32x8d_1117-1534-all',
           'FPN_resnet34_1116-2302-all',
           'FPN_dpn92_1117-0102-all',
           'FPN_densenet161_1117-0914-all',
           ]

for fold in folders:
    files = os.listdir(os.path.join(SAVE_PATH,'segment',fold))
    if 'cloud_mask.npy' in files or 'seg_probs.npy' in files:
        inferenced = True
    else:
        inferenced = False

    with open(os.path.join(SAVE_PATH,'segment',fold,'config'), 'r') as f:
        loss = 0
        for line in f.readlines():
            if 'Validate Loss' in line:
                loss += float(line.strip().split(':')[-1])
        loss /= 5
        print('{}\t\t{:.4f}\t{:6}'.format(fold,loss,inferenced))

    # with open(os.path.join(SAVE_PATH,'segment',fold,'config.pkl'), 'rb') as f:
    #     config = pickle.load(f)
    # print(config['ENCODER'],config['encoder_weights'])
    # print(config.keys())
    # break
    # config['encoder_weights'] = encoder_weights_dict[config['ENCODER']]
    # with open(os.path.join(SAVE_PATH,'segment',fold,'config.pkl'), 'wb') as f:
    #     pickle.dump(config,f)