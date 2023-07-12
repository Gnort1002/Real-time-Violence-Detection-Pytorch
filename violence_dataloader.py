import torch
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import config

class Dataset(data.Dataset):
    def __init__(self, data_list=[], skip_frame=1, time_step=30):
        self.time_step = time_step
        self.skip_frame = skip_frame
        self.data_list = self._build_data_list(data_list)

    def __len__(self):
        return len(self.data_list) // self.time_step