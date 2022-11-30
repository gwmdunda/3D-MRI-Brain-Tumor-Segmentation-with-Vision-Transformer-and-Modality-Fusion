from monai.losses import DiceLoss, DiceCELoss
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from models import UNETR, ViTSegmentation
import torch

import datetime

#Transform configs
PIXDIM = (1.0, 1.0, 1.0)
IMG_SIZE = (128, 128, 128)
FLIP_PROB = 0.5

DEVICE = 'cuda'
SEED = 0

#Model related configs
MODEL = ViTSegmentation
# MODEL_CONFIGS = {"in_channels" : 4, #Number of modalities
# "out_channels" : 3, #TC, WT, ET
# "feature_size" : 16,
# "hidden_size" : 768,
# "mlp_dim" : 3072,
# "num_heads" : 2,
# "pos_embed" : "perceptron",
# "norm_name" : "instance",
# "res_block" : True,
# "dropout_rate" : 0.0}

MODEL_CONFIGS ={
"spatial_dims": 3,
"in_channels": 4,
"out_channels": 3,
"img_size": IMG_SIZE,
"patch_size": 16,
"hidden_size": 768,
"mlp_dim": 3072
}

#Contribution-related model configs
SEPARATE_PATCH_EMBED = True #False for standard UNETR implementation, True for our implementation
DIFFERENT_PATCH_EMBED = False #False for using the same patch projection and True for different patch projection for each modality
AGG_METHOD = "joint_conv_decoder" #standard: None, embedding fusion: "joint_conv", encoder fusion "joint_conv_encoder" decoder fusion "joint_conv_decoder"

#Training configs
TRAIN_BATCH_SIZE = 4
VAL_BATCH_SIZE = 1
MAX_ITERATIONS = 160

LOSS_FN_CLASS = DiceLoss
LOSS_FN_CONFIG = {"smooth_nr":0, "smooth_dr":1e-5, "squared_pred":True, "to_onehot_y":False, "sigmoid":True}
OPTIMIZER = torch.optim.AdamW
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
OPTIMIZER_CONFIG = {"lr":LEARNING_RATE, "weight_decay":WEIGHT_DECAY}
LR_SCHEDULER = torch.optim.lr_scheduler.CosineAnnealingLR
LR_SCHEDULER_CONFIG = {"T_max": MAX_ITERATIONS}


#Naming configs
EXP_NAME="ViTAutoEncoder"
TRAIN_DATASET_PATH = 'BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/'
TEST_DATASET_PATH = 'BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData'
EXP_PATH = f"Exp/{EXP_NAME}"
SAVED_MODEL_NAME = f"results/trained_models/{EXP_NAME}_{MODEL.__name__}_best_{str(datetime.datetime.today())}.pth"
LOAD_MODEL_NAME = "results/trained_models/UNETR_best.pth"
GRAPH_PATH = f"results/graphs/{EXP_NAME}_{str(datetime.datetime.today())}"
METRIC_PATH = f"results/metrics/{EXP_NAME}_{str(datetime.datetime.today())}.txt"
SAMPLE_PATH = f"results/samples/{EXP_NAME}"
WITH_SAMPLE = True #Whether showing samples are required
