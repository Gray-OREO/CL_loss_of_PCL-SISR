import torch
import random
import cv2
import numpy as np


def get_neg_samples(hr_im, ne_num=4):
    '''
    Design the number of negative examples by the parameter 'ne_num', and change the operator in line 14-19 for your own strategy.
    hr_im: [batch_size, channel, height, width] tensor
    negsamps: [batch_size, ne_num, channel, height, width] tensor
    '''
    B, C, H, W = hr_im.shape
    negsamps = torch.zeros(B, ne_num, C, H, W)
    for b, img_hr__ in enumerate(hr_im):
        img_hr__ = img_hr__.detach().cpu().numpy()
        for k in range(ne_num):  
            min_size = 11
            addition = random.choice((0, 2, 4, 6, 8, 10, 12))
            size = min_size + addition
            kernel_size = (size, size)
            neg_samp = cv2.GaussianBlur(img_hr__, kernel_size, 0)
            neg_samp = torch.from_numpy(neg_samp)
            negsamps[b] = neg_samp
    return negsamps


def get_pos_samples(img_hr, pe_num=1):
    '''
    Similar to get_neg_samples.
    '''
    B, C, H, W = img_hr.shape
    possamp = torch.zeros(B, pe_num, C, H, W)
    img_hr = img_hr.detach().cpu().numpy()
    for m, img_hr_ in enumerate(img_hr):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
        pos_samp = cv2.filter2D(img_hr_, -1, kernel)
        pos_samp = torch.tensor(pos_samp).unsqueeze(0)
        possamp[m] = pos_samp
    return possamp


if __name__ == '__main__':
    inp = torch.randn(8, 3, 255, 255)
    out1 = get_neg_samples(inp)
    out2 = get_pos_samples(inp)
    print(out1.shape)
    print(out2.shape)
