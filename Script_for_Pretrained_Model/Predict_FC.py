import os
import torch
from PIL import Image
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import shutil
import time
import cv2
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model

size = (256, 256)
# Set cpu or GPU
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")
# Load Model
# model_path=r'checkpoints/GAN_Control_Ensemble_pix2pix_aligned_Resnet256/200_net_G.pth'
model_path=r"F:\pix2pix_FC\checkpoints\pix2pix_FC\295_net_G.pth"

# starttime_load = time.time()
if '_jit.pt' in model_path:
    GAN_generator = torch.jit.load(model_path, map_location='cpu').to(device)
else:
    GAN_generator = torch.load(model_path, map_location='cpu').to(device)
    if isinstance(GAN_generator,nn.DataParallel):
        GAN_generator=GAN_generator.module

GAN_generator.eval()
GAN_generator=GAN_generator.to(device)
# endtime_load = time.time()
print('model:',GAN_generator)
# print('loadtime:',round(endtime_load-starttime_load,2))


# ---Calculate parameters---
# from fvcore.nn import FlopCountAnalysis, parameter_count_table
# Parameters = parameter_count_table(GAN_generator)
# print('Parameters:', Parameters)

#
# def print_model_parm_nums(model):
#     total = sum([param.nelement() for param in model.parameters()])
#     print('  + Number of params: %.2fM' % (total / 1e6))
#
# Parameters_sum=print_model_parm_nums(model.netG)

def get_transform():
    transform_list = []
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)
# Preprocessing
G_transform = get_transform()

def picture_strengthen(file_path,GAN_generator,device):
    # Read images
    test_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), 1)[:, :, ::-1]
    # Copy original data
    test_img_ori = test_img.copy()
    # convert grey scale to RGB
    test_img = Image.fromarray(test_img.astype('uint8')).convert('RGB')
    # Unify the size of images
    img = test_img.resize(size)
    # img = transform_strengthen(img)
    # preprocessing
    img = G_transform(img)

    # remove 0 chanel
    img = img.unsqueeze(0).to(device)
    #  forward propagation
    if '-' in file_path:
        file_path=file_path.replace('-','')
    A,B,VF = os.path.split(file_path)[-1][:-4].split('_')[:3]
    AR = float(A)/float(B)
    VF = float(VF)
    GAN_generator=GAN_generator.to(device)
    generated = GAN_generator.forward(img, torch.Tensor([AR]).to(device)/4.0, torch.Tensor([VF]).to(device))
    # read image
    image_numpy = generated.data[0].cpu().float().numpy()
    # transpose chanel
    image_numpy=np.transpose(image_numpy, (1, 2, 0))
    # resolution +1，divided by 2.0，then time 255
    image_numpy = (image_numpy+ 1) / 2.0 * 255.0
    # limit the range of values in 0-255
    image_numpy = np.clip(image_numpy, 0, 255)
    # change the type of data to uint8
    image_numpy = image_numpy.astype(np.uint8)
    #
    image_numpy=cv2.resize(image_numpy,(test_img_ori.shape[1],test_img_ori.shape[0]))


    format=os.path.splitext(file_path)[-1]
    # save_file_path = file_path.replace(format, "_strengthen" + format)
    save_file_path = os.path.join(save_dir,os.path.split(file_path)[-1])
    cv2.imencode(format, image_numpy[:,:,::-1])[1].tofile(save_file_path)
    # ------------------------------save end----------------------------------
    # ------------------------------for test ---------------------------------
    # format = os.path.splitext(file_path)[-1]
    # save_file_path = os.path.join(save_dir, os.path.split(file_path)[-1])
    # cv2.imencode(format, final_img[:, :, ::-1])[1].tofile(save_file_path)
    # ------------------------------save end----------------------------------

if __name__=='__main__':
    src_dir=r"D:\GAN\pytorch-CycleGAN-and-pix2pix\datasets\half_dataset\test"
    save_dir=src_dir+'_pix2pix_FC_predicts'.format(os.path.split(model_path)[-1][:4])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_list=os.listdir(src_dir)

    for file in file_list:
        file_path = os.path.join(src_dir, file)
        starttime = time.time()

        picture_strengthen(file_path,GAN_generator,device=device)
        endtime = time.time()
        print(file,'_time:', round(endtime - starttime,2))
    print('save_dir:',save_dir)

