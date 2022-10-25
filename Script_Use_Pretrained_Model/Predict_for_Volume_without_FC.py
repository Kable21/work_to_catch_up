import os
import torch
from PIL import Image
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
import shutil
import time
import cv2
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model

size = (256, 256)
# Set cpu或GPU
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device('cpu')
# Load Models
# model_path=r'400_net_G_3_512_512_jit.pt'
# ---------------**********************8_________________
# model_path=r'Ensemble_pix2pix_ori/200_net_G_3_256_256_jit.pt'
model_path=r'checkpoints/pix2pix_combined_volfrac256/150_net_G.pth'
opt = TrainOptions().parse()   # get training options

if '_jit.pt' in model_path:
    GAN_generator = torch.jit.load(model_path, map_location='cpu').to(device)
else:
    GAN_generator = torch.load(model_path, map_location='cpu').to(device)

GAN_generator.eval()
print('model:',GAN_generator)


def get_transform():
    transform_list = []
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

# Preprocessing
G_transform = get_transform()

def picture_strengthen(file_path):
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
    #  forward transfer
    generated = GAN_generator.forward(img)
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
    src_dir=r'datasets/Change of Volfrac/input/Chang of Volfrac_TO_test'
    save_dir=src_dir+'_Ori_pix2pix_predicts_epoch_epoch_150' #{}'.format(os.path.split(model_path)[-1][:3])'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_list=os.listdir(src_dir)
    for file in file_list:
        file_path=os.path.join(src_dir,file)
        picture_strengthen(file_path)
    print('save_dir:',save_dir)

