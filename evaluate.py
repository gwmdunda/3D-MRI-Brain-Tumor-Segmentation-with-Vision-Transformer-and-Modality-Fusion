import os
import cv2
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

from dataset import BraTS2020Dataset, ConvertToMultiChannelBasedOnBratsClassesd
from config import *
from data_transform import get_transforms

from monai.data import DataLoader
from monai.metrics import DiceMetric
from monai.data import  decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import (Compose, Activations, AsDiscrete)

def save_segmentation_results(ground_truth, original_img, prediction, root_dir, idx):
    gt_core = ground_truth[0,90,:,:]
    gt_wt = ground_truth[1,90,:,:]
    gt_et = ground_truth[2,90,:,:]
    original_img = original_img[0,90,:,:]

    core = prediction[0,:,:,:]
    wt = prediction[1,:,:,:]
    et = prediction[2,:,:,:]

    f, axarr = plt.subplots(1, 5, figsize = (18, 50))
    for i in range(5): # for each image, add brain background
        axarr[i].imshow(cv2.resize(original_img, (128, 128)), cmap="gray", interpolation='none')
    
    axarr[0].imshow(cv2.resize(original_img, (128, 128)), cmap="gray")
    axarr[0].title.set_text('Original image flair')
    axarr[0].set_axis_off()
    axarr[1].imshow(gt_core, cmap="Reds", interpolation='none', alpha=0.3) # ,alpha=0.3,cmap='Reds'
    axarr[1].title.set_text('Ground truth tumor core')
    axarr[1].set_axis_off()
    axarr[2].imshow(core[90,:,:], cmap="OrRd", interpolation='none', alpha=0.3)
    axarr[2].title.set_text(f'Predicted tumor core')
    axarr[2].set_axis_off()
    axarr[3].imshow(gt_wt, cmap="Reds", interpolation='none', alpha=0.3) # ,alpha=0.3,cmap='Reds'
    axarr[3].title.set_text('Ground truth whole tumor')
    axarr[3].set_axis_off()
    axarr[4].imshow(wt[90,:,], cmap="OrRd", interpolation='none', alpha=0.3)
    axarr[4].title.set_text(f'Predicted whole tumor')
    axarr[4].set_axis_off()
    
    
    f.savefig(os.path.join(root_dir, f"sample_{idx}.pdf"), bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close()


def pathListIntoIds(dirList):
    x = []
    for i in range(0,len(dirList)):
        x.append(dirList[i][dirList[i].rfind('/')+1:])
    return x
post_pred = Compose(
    [Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
    )

test_directories = [f.path for f in os.scandir(TEST_DATASET_PATH) if f.is_dir()]
test_directories.sort()

test_ids = pathListIntoIds(test_directories)
_, test_transform = get_transforms()
test_ds = BraTS2020Dataset(test_ids, TEST_DATASET_PATH, test_transform)

test_loader = DataLoader(test_ds, batch_size=1)

model = MODEL(
    **MODEL_CONFIGS
    ).to(DEVICE)
model.load_state_dict(torch.load(LOAD_MODEL_NAME))
model.eval()
dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

if not os.path.exists(SAMPLE_PATH):
    os.makedirs(SAMPLE_PATH)

with torch.no_grad():
    for step, batch in enumerate(test_loader):
        val_inputs, val_labels = (batch["img"].to(DEVICE), batch["gt"].to(DEVICE))
        val_outputs = sliding_window_inference(val_inputs, IMG_SIZE, 1, model, overlap = 0.5)

        val_outputs_list = decollate_batch(val_outputs)
        val_output_convert = [
            post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
        ]
        dice_metric(y_pred=val_output_convert, y=val_labels)
        dice_metric_batch(y_pred=val_output_convert, y=val_labels)
        if WITH_SAMPLE:
            save_segmentation_results(batch["gt"][0].cpu().numpy(), batch["orig_img"][0].cpu().numpy(), val_output_convert[0].cpu().numpy(), SAMPLE_PATH, step+1)
    mean_dice_val = dice_metric.aggregate().item()
    dice_metric_batch_agg = dice_metric_batch.aggregate()

    mean_dice_metric_tc = dice_metric_batch_agg[0].item()
    mean_dice_metric_wt = dice_metric_batch_agg[1].item()
    mean_dice_metric_et = dice_metric_batch_agg[2].item()
    with open(METRIC_PATH, 'w') as f:
        f.write(f"Mean dice coeff on test set: {mean_dice_val} \n")
        f.write(f"TC dice coeff on test set: {mean_dice_metric_tc} \n")
        f.write(f"WT dice coeff on test set: {mean_dice_metric_wt} \n")
        f.write(f"ET dice coeff on test set: {mean_dice_metric_et}")
        