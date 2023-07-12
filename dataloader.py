import torch
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import config

class Dataset(data.Dataset):
    def __init__(self, data_list=[], skip_frame=1, time_step=30):
        '''
        定义一个数据集，从UCF101中读取数据
        Define a dataset that reads data from UCF101
        '''
        # 用来将类别转换为one-hot数据 
        # Used to convert categories to one-hot data
        self.labels = []
        # 用来缓存图片数据，直接加载到内存中
        # Used to cache image data, loaded directly into memory
        self.images = []
        # 是否直接加载至内存中，可以加快训练速
        # Whether to load directly into memory, which can speed up training
        self.use_mem = False

        self.skip_frame = skip_frame
        self.time_step = time_step
        self.data_list = self._build_data_list(data_list)

    def __len__(self):
        return len(self.data_list) // self.time_step

    def __getitem__(self, index):
        # 每次读取time_step帧图片
        # Read time_step frames of pictures each time
        index = index * self.time_step
        imgs = self.data_list[index:index + self.time_step]

        # 图片读取来源，如果设置了内存加速，则从内存中读取
        # Picture reading source, if memory acceleration is set, read from memory
        if self.use_mem:
            X = [self.images[x[3]] for x in imgs]
        else:
            X = [self._read_img_and_transform(x[2]) for x in imgs]

        # 转换成tensor
        # Convert to tensor
        X = torch.stack(X, dim=0)

        # 为这些图片指定类别标签
        # Specify category labels for these pictures
        y = torch.tensor(self._label_category(imgs[0][0]))
        return X, y

    def transform(self, img):
        return transforms.Compose([
            transforms.Resize((config.img_w, config.img_h)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])(img)

    def _read_img_and_transform(self, img:str):
        return self.transform(Image.open(img).convert('RGB'))

    def _build_data_list(self, data_list=[]):
        '''
        构建数据集
        '''
        if len(data_list) == 0:
            return []

        data_group = {}
        for x in tqdm(data_list, desc='Building dataset'):
            # 将视频分别按照classname和videoname分组
            # Group videos by classname and videoname respectively
            [classname, videoname] = x[0:2]
            if classname not in data_group:
                data_group[classname] = {}
            if videoname not in data_group[classname]:
                data_group[classname][videoname] = []

            # 将图片数据加载到内存
            # Load image data into memory
            if self.use_mem:
                self.images.append(self._read_img_and_transform(x[2]))

            data_group[classname][videoname].append(list(x) + [len(self.images) - 1])

        # 处理类别变量
        # Process category variables
        self.labels = list(data_group.keys())

        ret_list = []
        n = 0

        # 填充数据
        # Fill data
        for classname in data_group:
            video_group = data_group[classname]
            for videoname in video_group:
                # 如果某个视频的帧总数没法被time_step整除，那么需要按照最后一帧进行填充
                # If the total number of frames of a video cannot be divided by time_step, then it needs to be filled according to the last frame
                video_pad_count = len(video_group[videoname]) % self.time_step #20
                video_group[videoname] += [video_group[videoname][-1]] * (self.time_step - video_pad_count)
                ret_list += video_group[videoname]
                n += len(video_group[videoname])

        return ret_list

    def _label_one_hot(self, label):
        '''
        将标签转换成one-hot形式
        '''
        if label not in self.labels:
            raise RuntimeError('不存在的label！')
        one_hot = [0] * len(self.labels)
        one_hot[self.labels.index(label)] = 1
        return one_hot

    def _label_category(self, label):
        '''
        将标签转换成整型
        '''
        if label not in self.labels:
            raise RuntimeError('不存在的label！')
        c_label = self.labels.index(label)
        return c_label
