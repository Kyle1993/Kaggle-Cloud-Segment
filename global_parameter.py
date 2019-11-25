# DATA_PATH = '/media/kyle/Data/Data/cloud_segment'
# DATA_PATH = '/home/noel/pythonproject/clouds_segment/data'
DATA_PATH = '/home/jianglb/pythonproject/cloud_segment/data'
SAVE_PATH = 'checkpoints'


# TRAIN_NUM = 5546
BAD_IMG = ['046586a.jpg', '1588d4c.jpg', '1e40a05.jpg', '41f92e5.jpg', '449b792.jpg', '563fc48.jpg', '8bd81ce.jpg',
           'c0306e5.jpg', 'c26c635.jpg', 'e04fea3.jpg', 'e5f2f24.jpg', 'eda52f2.jpg', 'fa645da.jpg']
TRAIN_NUM = 5533 # 5546 - 13
TEST_NUM = 3698
IMAGE_SHAPE = (1400, 2100, 3)
IMAGE_SIZE = (1400, 2100)
STANDARD_RESIZE = (350, 525)
LABELS = ['Fish', 'Flower', 'Gravel', 'Sugar']
LABEL_NUM = len(LABELS)

# MEAN = (0.485, 0.456, 0.406)
# STD = (0.229, 0.224, 0.225)

K = 5
FOLD_NUM = [1106, 1106, 1106, 1106, 1109]

RESIZE = (480, 640)# (320, 512)24, (352, 544)24, (480, 640)16, (640, 960)8, (960, 1408)4 ,
LR = 5e-4
WD = 1e-5
BATCH_SIZE = 16
NUM_EPOCH = 15

encoder_weights_dict = {
    'resnet18': 'imagenet',
    'resnet34': 'imagenet',
    'resnet50': 'imagenet',
    'resnet101': 'imagenet',
    'densenet161': 'imagenet',
    'densenet169': 'imagenet',
    'densenet201': 'imagenet',
    'efficientnet-b0': 'imagenet',
    'efficientnet-b3': 'imagenet',
    'efficientnet-b5': 'imagenet',

    'dpn92':'imagenet+5k',
    'dpn68b':'imagenet+5k',

    'resnext101_32x8d':'instagram',

         }

# temp
FOLD = 0
VALIDATE_STEP = 10
MODEL = None
ENCODER = None
CLASSIFER = None
encoder_weights = None
dropout = 0.4
ED_drop = 0.2
early_stop = 4
patience = 1
loss_ratio = 0.25

milestones = [10,14,16,18,20]
gamma = 0.2


