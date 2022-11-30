import os
import nibabel as nib
import numpy as np

import torch

from monai.data import Dataset
from monai.transforms import MapTransform

class BraTS2020Dataset(Dataset):
    def __init__(self, list_IDs, path, transform):
        self.list_IDs = list_IDs
        self.transform = transform
        self.path = path

    def __getitem__(self, idx):
        item = {'img': None, 'gt': None}

        i = self.list_IDs[idx]
        case_path = os.path.join(self.path, i)

        data_path = os.path.join(case_path, f'{i}_flair.nii')
        flair = nib.load(data_path).get_fdata()   

        data_path = os.path.join(case_path, f'{i}_t1ce.nii')
        ce = nib.load(data_path).get_fdata()

        data_path = os.path.join(case_path, f'{i}_t1.nii')
        t1 = nib.load(data_path).get_fdata()

        data_path = os.path.join(case_path, f'{i}_t2.nii')
        t2 = nib.load(data_path).get_fdata()
        
        data_path = os.path.join(case_path, f'{i}_seg.nii')
        seg = nib.load(data_path).get_fdata()

        seg[seg==4] = 3

        item['img'] = np.vstack((np.expand_dims(flair, axis=0), np.expand_dims(ce, axis=0), np.expand_dims(t1, axis=0), np.expand_dims(t2, axis=0))).astype(np.float32)
        item['gt'] = seg.astype(np.float32)

        if self.transform:
            item = self.transform(item)
        
        return item
    
    def __len__(self):
        return len(self.list_IDs)

class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the necrotic and non-enhancing tumor core
    label 2 is the the peritumoral edema
    label 3 is the GD-enhancing tumor 
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 1 and label 3 to construct TC
            result.append(torch.logical_or(d[key] == 1, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(
                torch.logical_or(
                    torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1
                )
            )
            # label 2 is ET
            result.append(d[key] == 3)
            d[key] = torch.stack(result, axis=0).float()
        return d