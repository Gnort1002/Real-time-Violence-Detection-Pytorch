import torch
from torch import nn
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from PIL import Image
import pandas
import os
import argparse
import cv2

from dataloader import Dataset
from model import CNNEncoder, RNNDecoder
import config


def _infer(checkpoint: str, skip_frames=3, time_step=50)->list:
    """Inference the model and return the labels.

    Args:
        checkpoint(str): The checkpoint where the model restore from.
        path(str): The path of videos.
        labels(list): Labels of videos.

    Returns:
        A list of labels of the videos.
    """
    images = []
    print('Loading model from {}'.format(checkpoint))
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Build model
    model = nn.Sequential(
        CNNEncoder(**config.cnn_encoder_params),
        RNNDecoder(**config.rnn_decoder_params)
    )
    model.to(device)
    model.eval()

    # Load model
    ckpt = torch.load(checkpoint)
    model.load_state_dict(ckpt['model_state_dict'])
    print('Model has been loaded from {}'.format(checkpoint))

    label_map = [-1] * config.rnn_decoder_params['num_classes']
    # load label map
    if 'label_map' in ckpt:
        label_map = ckpt['label_map']

    vid = cv2.VideoCapture(0)

    count_frame = 0

    while(True):

        # font
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # org
        org = (50, 50)
        
        # fontScale
        fontScale = 5
        
        # Blue color in BGR
        color = (255, 0, 0)
        
        # Line thickness of 2 px
        thickness = 2
        # Capture the video frame
        # by frame

        ret, frame = vid.read()
    
        # Display the resulting frame
        count_frame += 1

        if count_frame % skip_frames == 0:
            images.append(Image.fromarray(frame))
        
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if len(images) == time_step:
        # Do inference
            with torch.no_grad():
                # read images from video
                # apply transform
                images = [Dataset.transform(None, img) for img in images]
                # stack to tensor, batch size = 1
                images = torch.stack(images, dim=0).unsqueeze(0)
                # do inference
                images = images.to(device)

                pred_y = model(images) # type: torch.Tensor
                pred_y = pred_y.argmax(dim=1).cpu().numpy().tolist()

                print(pred_y[0])
            images = images.tolist()
            images = images[int(time_step/2):time_step]    

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    return

def parse_args():
    parser = argparse.ArgumentParser(usage='python3 inference.py -r path/to/checkpoint')
    parser.add_argument('-r', '--checkpoint', help='path to the checkpoint')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    _infer(args.checkpoint)
