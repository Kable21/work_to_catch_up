from PIL import Image
import os
import re
import time
import sys
from multiprocessing import Pool
import threading
import cv2
import numpy as np

def rewrite(input_dir,filename,save_dir,save_format,size,save_name):
    img = Image.open(os.path.join(input_dir,filename))
    if img.mode=='RGBA':
        img= img.convert("RGB")
    #
    width, height=img.width,img.height   # (100,50 )
    img = img.resize((size[0], int(height* size[0]/width)))   # 512
    width, height=img.width,img.height
    img=np.array(img)   # (256,512)
    img_new = np.zeros((size))
    img_new[int((size[1]-height)/2) : int(size[1]-(size[1]-height)/2),:] =img
    cv2.imwrite(os.path.join(save_dir,save_name+save_format),img_new)
    # img_new=Image.fromarray(img_new).convert("L")
    # img_new.save(os.path.join(save_dir,save_name+save_format))

'''
add black padding to top&bottom

'''

if __name__ == '__main__':
    # set number of parellel precessors
    processor = 4
    res = []
    # creat pool
    p = Pool(processor)
    src_dir = r'E:\Projects_Cooperation\pix2pix_template\datasets\GAN_Control_Ensemble_ori\targetB'
    save_dir = src_dir+'_rename_256'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # set saved image size and format 
    size=(256,256)  # (w,h)
    save_format = '.png'
    filenames = os.listdir(src_dir)
    # record start time
    t_start = time.time()
    for i, filename in enumerate(filenames):
        print('{}/{}'.format(i, len(filenames)))
        
        #num_str=str(int(filename.split('.')[0]))
        # if '_AM' in filename:
        #     save_name=filename.replace('_AM', '')

        if 'B' in src_dir:
            save_name=filename[:-4].replace('AM_', '').replace('_B','')+'_B'
        else:
            save_name=filename[:-4].replace('_A','')+'_A'
        if '-0' in save_name:
            save_name=save_name.replace('-0','_0')
        rewrite(src_dir, filename, save_dir, save_format, size, save_name)
        #res.append(p.apply_async(rewrite, args=(src_dir, filename, save_dir, save_format, size, save_name)))
        print(str(i) + ' processor started !')
    
    for i in res:
        print(i.get())  



