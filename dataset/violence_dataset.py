import os
import pandas as pd
import numpy as np
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset 
import sys

sys.path.insert(0, '/home/gnort/Work/Realtime-Violence-Detection-Pytorch/utils')

from capture import capture_frame

class ViolenceDataset(Dataset):

    def __init__(self, datas, timesep=30,rgb=3,h=120,w=120):
        """
        Args:
            datas: pandas dataframe contain path to videos files with label of them
            timesep: number of frames
            rgb: number of color channels
            h: height
            w: width
                 
        """
        self.dataloctions = datas
        self.timesep, self.rgb, self.h, self.w = timesep, rgb, h, w
        
    def __len__(self):
        return len(self.dataloctions)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        video = capture_frame(self.dataloctions.iloc[idx, 0],self.timesep,self.rgb,self.h,self.w)
        sample = {'video': torch.from_numpy(video), 'label': torch.from_numpy(np.asarray(self.dataloctions.iloc[idx, 1]))}

        return sample