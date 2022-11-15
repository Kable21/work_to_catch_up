import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
from PIL import Image
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
import shutil
import time
import torch.nn as nn
import cv2
from scipy.stats import pearsonr
from sklearn import metrics

def print_log(print_string, log_file, visible=True):
    if visible:
        print("{}".format(print_string))
    # write in log file
    log_file.write('{}\n'.format(print_string))
    # log the data into the file
    log_file.flush()

size = (256, 256)
# set device as cpu or GPU
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")
save_dir=r'results_Ours'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)



# ---calculate size of parameters---
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
# preprocessing
G_transform = get_transform()

# def picture_strengthen(GAN_generator, file_path, device):
#     # Read images
#     test_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), 1)[:, :, ::-1]
#     # Copy original data
#     test_img_ori = test_img.copy()
#     # convert grey scale to RGB
#     test_img = Image.fromarray(test_img.astype('uint8')).convert('RGB')
#     # resize the image
#     img = test_img.resize(size)
#     # preprocessing  []
#
#     img = G_transform(img)
#     # remove 0 chanel
#     img = img.unsqueeze(0).to(device)
#     # forward propagation
#     if '-' in file_path:
#         file_path=file_path.replace('-','')
#     A,B,VR=os.path.split(file_path)[-1][:-4].split('_')[:3]
#     AP= float(A)/float(B)
#     volume = float(VR)
#
#
#     # generated = GAN_generator.forward(img)
#
#     if '-' in file_path:
#         file_path = file_path.replace('-', '')
#     A, B, VF = os.path.split(file_path)[-1][:-4].split('_')[:3]
#     AR = float(A) / float(B)
#     VF = float(VF)
#     # GAN_generator = GAN_generator.to(device)
#     generated = GAN_generator.forward(img, torch.Tensor([AR]).to(device) / 4.0, torch.Tensor([VF]).to(device))
#
#     # get the matrix of image
#     image_numpy = generated.data[0].cpu().float().numpy()
#     # transpose chanel
#     image_numpy=np.transpose(image_numpy, (1, 2, 0))
#     # resolution +1，divided by 2.0，then time 255
#     image_numpy = (image_numpy+ 1) / 2.0 * 255.0
#     # limit the range of values in 0-255
#     image_numpy = np.clip(image_numpy, 0, 255)
#     # change the type of data to uint8
#     image_numpy = image_numpy.astype(np.uint8)
#     #
#     image_numpy=cv2.resize(image_numpy,(test_img_ori.shape[1],test_img_ori.shape[0]))
#
#     # save results
#     # format=os.path.splitext(file_path)[-1]
#     # # save_file_path = file_path.replace(format, "_strengthen" + format)
#     # save_file_path = os.path.join(save_dir,os.path.split(file_path)[-1])
#     # cv2.imencode(format, image_numpy[:,:,::-1])[1].tofile(save_file_path)
#     return image_numpy
#     # ------------------------------save end----------------------------------
#     # ------------------------------for test ---------------------------------
#     # format = os.path.splitext(file_path)[-1]
#     # save_file_path = os.path.join(save_dir, os.path.split(file_path)[-1])
#     # cv2.imencode(format, final_img[:, :, ::-1])[1].tofile(save_file_path)
#     # ------------------------------save end----------------------------------
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
    # for original pix2pix
    generated = GAN_generator.forward(img)
    # for G with FC
    # generated = GAN_generator.forward(img, torch.Tensor([AR]).to(device)/4.0, torch.Tensor([VF]).to(device))
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
    return image_numpy
if __name__=='__main__':
    src_dir=r'D:\GAN\pytorch-CycleGAN-and-pix2pix\datasets\half_dataset\test'
    target_dir=r'D:\GAN\pytorch-CycleGAN-and-pix2pix\datasets\half_dataset\test_target'
    model_dir=r"F:\models\checkpoints\pix2pix"
    #save_dir=src_dir+'_pix2pix_Ours_predicts_catt_{}'.format(os.path.split(model_path)[-1][:4])
    log=open(os.path.join(model_dir,'log_MAE.txt'),'w',encoding='UTF-8')
    print_log('model_dir:{}'.format(model_dir), log)
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    file_list=os.listdir(src_dir)
    MAE_all=[]
    RMSE_all=[]
    pearson_all=[]
    # load model
    models=os.listdir(model_dir)
    model_list = []
    for file in models:
        # print(file)
        if '_net_G.pth' in file:
            model_list.append(file)
    print(model_list)
    if 'GAN_Control_Ensemble_pix2pix_aligned_Resnet_catt_256_jit.zip' in model_list:
        model_list.remove('GAN_Control_Ensemble_pix2pix_aligned_Resnet_catt_256_jit.zip')
    if 'latest_net_G.pth' in model_list:
        model_list.remove('latest_net_G.pth')
    if 'log_MAE.txt' in model_list:
        model_list.remove('log_MAE.txt')
    # if 'loss_log.txt' in model_list:
    #     model_list.remove('loss_log.txt')
    # if 'train_opt.txt' in model_list:
    #     model_list.remove('train_opt.txt')
    # if 'web' in model_list:
    #     model_list.remove('web')

    model_list.sort(key=lambda x:int(x.split('_')[0]))

    for model_file in model_list:
        try:
            model_path = os.path.join(model_dir,model_file)
            # model_path=r"F:\pix2pix_attention\checkpoints\pix2pix_FC\295_net_G.pth"
            if '_jit.pt' in model_path:
                GAN_generator = torch.jit.load(model_path, map_location='cpu').to(device)
            else:
                GAN_generator = torch.load(model_path, map_location='cpu').to(device)
                if isinstance(GAN_generator, nn.DataParallel):
                    GAN_generator = GAN_generator.module

            GAN_generator.eval()

            # print('model:', GAN_generator)
            with torch.no_grad():
                for idx,file in enumerate(file_list):
                    print('{}/{}'.format(idx, len(file_list)))
                    # predict results
                    file_path=os.path.join(src_dir,file)
                    predict_img=picture_strengthen(file_path=file_path,GAN_generator=GAN_generator,device=device)
                    # GT
                    target=cv2.imread(os.path.join(target_dir,file))
                    # print('predict_img:{},target:{}'.format(predict_img.shape,target.shape))

                    y_true=target[:,:,0].reshape(-1)/255.0
                    y_true[y_true>0.5]=1.0
                    y_true[y_true<=0.5]=0
                    y_pred=predict_img[:,:,0].reshape(-1)/255.0
                    y_pred[y_pred>0.5]=1.0
                    y_pred[y_pred<=0.5]=0.0

                    # MAE，RMSE，Pearson
                    MAE = np.mean(np.abs(y_true - y_pred))
                    MAE_all.append(MAE)
                    MAE_2 = metrics.mean_absolute_error(y_true, y_pred)

                    RMSE = np.sqrt(np.mean(np.square(y_true - y_pred)))
                    RMSE_all.append(RMSE)

                    RMSE_2 = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
                    pearson, ttest = pearsonr(y_true, y_pred)
                    pearson_all.append(pearson)
        except:
            print('Error Model:',model_path)
        MAE_mean = np.mean(np.array(MAE_all))
        RMSE_mean = np.mean(np.array(RMSE_all))
        pearson_mean= np.mean(np.array(pearson_all))
        #  print values
        print_log('Epoch:{},MAE:{:.5f}, RMSE:{:.5f}, pearson:{:.5f}'.format(model_file.split('_')[0],MAE_mean,RMSE_mean , pearson_mean),log)
    print('save_dir:',model_dir)
    log.close()
