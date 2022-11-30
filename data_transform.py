from monai.transforms import (
    AsDiscrete,
    Compose,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    Spacingd,
    ToTensord,
    ResizeWithPadOrCropd,
    Activations
)
from dataset import ConvertToMultiChannelBasedOnBratsClassesd
from config import *

def get_transforms():
    train_transforms = Compose(
        [
            ToTensord(keys=["img", 'gt']),
            ConvertToMultiChannelBasedOnBratsClassesd(keys=["gt"]),
            Orientationd(keys=["img", "gt"], axcodes="RAS"),
            Spacingd(
            keys=["img", "gt"],
            pixdim=PIXDIM,
            mode=("bilinear", "nearest"),
            ), #Resample input image into the specified pixdim, pixdim is the scaling
            ResizeWithPadOrCropd(keys=["img", "gt"], spatial_size=IMG_SIZE),
            RandFlipd(keys=["img", "gt"], prob=FLIP_PROB, spatial_axis=0),
            RandFlipd(keys=["img", "gt"], prob=FLIP_PROB, spatial_axis=1),
            RandFlipd(keys=["img", "gt"], prob=FLIP_PROB, spatial_axis=2),
            NormalizeIntensityd(keys="img", nonzero=True, channel_wise=True),
        ])
    val_transforms = Compose(
        [
            ToTensord(keys=["img", 'gt']),
            ConvertToMultiChannelBasedOnBratsClassesd(keys=["gt"]),
            Orientationd(keys=["img", "gt"], axcodes="RAS"),
            Spacingd(
            keys=["img", "gt"],
            pixdim=PIXDIM,
            mode=("bilinear", "nearest"),
            ), #Resample input image into the specified pixdim, pixdim is the scaling
            ResizeWithPadOrCropd(keys=["img", "gt"], spatial_size=IMG_SIZE),
            NormalizeIntensityd(keys="img", nonzero=True, channel_wise=True),
        ]
    )
    return train_transforms, val_transforms