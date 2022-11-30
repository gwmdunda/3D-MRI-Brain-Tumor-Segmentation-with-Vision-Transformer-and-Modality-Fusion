from config import *
from dataset import BraTS2020Dataset, ConvertToMultiChannelBasedOnBratsClassesd
from data_transform import get_transforms

import os
from torch.utils.tensorboard import SummaryWriter
import torch
import warnings
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from monai.data import DataLoader
from monai.metrics import DiceMetric
from monai.data import  decollate_batch
from monai.utils import set_determinism
from monai.inferers import sliding_window_inference
from monai.transforms import (Compose, Activations, AsDiscrete)


SEGMENT_CLASSES = {
    0 : 'NOT tumor',
    1 : 'NECROTIC/CORE', # or NON-ENHANCING tumor CORE
    2 : 'EDEMA',
    3 : 'ENHANCING' # original 4 -> converted into 3 later
}

def pathListIntoIds(dirList):
    x = []
    for i in range(0,len(dirList)):
        x.append(dirList[i][dirList[i].rfind('/')+1:])
    return x



def validation(epoch_iterator_val, model, post_pred, dice_metric, dice_metric_batch, ):
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = (batch["img"].to(DEVICE), batch["gt"].to(DEVICE))
            val_outputs = sliding_window_inference(val_inputs, IMG_SIZE, 1, model, overlap = 0.5)
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [
                post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]
            dice_metric(y_pred=val_output_convert, y=val_labels)
            dice_metric_batch(y_pred=val_output_convert, y=val_labels)

        mean_dice_val = dice_metric.aggregate().item()
        dice_metric_batch_agg = dice_metric_batch.aggregate()

        mean_dice_metric_tc = dice_metric_batch_agg[0].item()
        mean_dice_metric_wt = dice_metric_batch_agg[1].item()
        mean_dice_metric_et = dice_metric_batch_agg[2].item()
        

        dice_metric.reset()
        dice_metric_batch.reset()
    return mean_dice_val, mean_dice_metric_tc, mean_dice_metric_wt, mean_dice_metric_et

def train(global_step, train_loader, dice_val_best, global_step_best, model, loss_fn, optimizer, lr_scheduler, writer, epoch_loss_values):
    model.train()
    epoch_loss = 0
    step = 0
    for step, batch in enumerate(train_loader):
        step += 1
        x, y = (batch["img"].to(DEVICE), batch["gt"].to(DEVICE))
        logit_map = model(x)
        if isinstance(logit_map, tuple):
            logit_map = logit_map[0]
        loss = loss_fn(logit_map, y)
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
    epoch_loss /= step
    lr_scheduler.step()
    global_step += 1
    writer.add_scalar('train_loss', epoch_loss, global_step)
    epoch_loss_values.append(epoch_loss)
    
    return global_step, dice_val_best, global_step_best

def main():
    train_and_val_directories = [f.path for f in os.scandir(TRAIN_DATASET_PATH) if f.is_dir()]
    train_and_val_directories.remove(TRAIN_DATASET_PATH+'BraTS20_Training_355')

    train_and_val_ids = pathListIntoIds(train_and_val_directories)
    train_test_ids, val_ids = train_test_split(train_and_val_ids,test_size=0.2)


    train_transforms, val_transforms = get_transforms()

    train_ds = BraTS2020Dataset(train_test_ids, TRAIN_DATASET_PATH,  train_transforms)
    val_ds = BraTS2020Dataset(val_ids, TRAIN_DATASET_PATH, val_transforms)
    

    model = MODEL(
    **MODEL_CONFIGS
    ).to(DEVICE)

    train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=VAL_BATCH_SIZE)

    loss_fn = LOSS_FN_CLASS(**LOSS_FN_CONFIG)
    optimizer = OPTIMIZER(model.parameters(), **OPTIMIZER_CONFIG)
    lr_scheduler = LR_SCHEDULER(optimizer, **LR_SCHEDULER_CONFIG)

    writer = SummaryWriter(log_dir=EXP_PATH)

    post_pred = Compose(
    [Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
    )

    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

    dice_val_best = 0.0
    global_step_best = 0
    global_step = 0
    epoch_loss_values = []
    metric_values = []
    dice_metric_values_tc = []
    dice_metric_values_wt = []
    dice_metric_values_et = []
    while global_step < MAX_ITERATIONS:
        global_step, dice_val_best, global_step_best = train(
            global_step, train_loader, dice_val_best, global_step_best, model, loss_fn, optimizer, lr_scheduler, writer, epoch_loss_values
        )
        dice_val, mean_dice_metric_tc, mean_dice_metric_wt, mean_dice_metric_et = validation(val_loader, model, post_pred, dice_metric, dice_metric_batch)
        dice_metric_values_tc.append(mean_dice_metric_tc)
        dice_metric_values_wt.append(mean_dice_metric_wt)
        dice_metric_values_et.append(mean_dice_metric_et)
        writer.add_scalar('val_dice', dice_val, global_step)
        writer.add_scalar('val_dice_tc', mean_dice_metric_tc, global_step)
        writer.add_scalar('val_dice_wt', mean_dice_metric_wt, global_step)
        writer.add_scalar('val_dice_et', mean_dice_metric_et, global_step)
        metric_values.append(dice_val)
        if dice_val > dice_val_best:
            dice_val_best = dice_val
            global_step_best = global_step
            torch.save(model.state_dict(), SAVED_MODEL_NAME)
            print(
                "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                    dice_val_best, dice_val)
            )
        else:
            print(
                "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                    dice_val_best, dice_val)
            )

    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 6))
    ax1.set_title("Training Dice Loss")
    x = [i + 1 for i in range(MAX_ITERATIONS)]
    y = epoch_loss_values
    ax1.set_xlabel("epoch")
    ax1.plot(x, y, color="red")

    ax2.set_title("Validation Mean Dice")
    y = metric_values
    ax2.set_xlabel("epoch")
    ax2.plot(x, y, color="green")
    fig.savefig(GRAPH_PATH+"_all.pdf",bbox_inches="tight")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (18, 6))
    ax1.set_title("Validation Mean Dice TC")
    y = dice_metric_values_tc
    ax1.set_xlabel("epoch")
    ax1.plot(x, y, color="blue")

    ax2.set_title("Validation Mean Dice WT")
    y = dice_metric_values_wt
    ax2.set_xlabel("epoch")
    ax2.plot(x, y, color="brown")

    ax3.set_title("Validation Mean Dice ET")
    y = dice_metric_values_et
    ax3.set_xlabel("epoch")
    ax3.plot(x, y, color="purple")
    fig.savefig(GRAPH_PATH+"_detail.pdf",bbox_inches="tight")


if __name__ == '__main__':
    set_determinism(seed=SEED)
    warnings.filterwarnings("ignore")
    main()