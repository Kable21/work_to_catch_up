#! -*- coding:utf-8 -*-
import os
import numpy as np
'''
Plot MAE，RMSE，Pearson

'''

# ------------------------------------ plot----------------------------
import matplotlib.pyplot as plt

# load results
# Epoch:5,MAE:0.11857, RMSE:0.34286, pearson:0.64998
log_path = r"F:\models\checkpoints\pix2pix\log_MAE.txt"
# log_path = r"F:\models\checkpoints\pix2pix\log_MAE.txt"

f=open(log_path,'r',encoding='UTF-8')
# -------------- '(epoch: 1, iters: 100, time: 0.015, data: 5.072) G_GAN: 1.005 G_L1: 21.786 D_real: 0.533 D_fake: 0.728' -------
MAE_all=[]
RMSE_all=[]
pearson_all=[]
Epoch_all=[]
bf_epoch=0

for line in f.readlines():
    line = line.strip()
    if 'Epoch:' in line:
        # Epoch:5,MAE:0.11857, RMSE:0.34286, pearson:0.64998
        Epoch, MAE, RMSE, pearson =line.split(',')
        Epoch=int(Epoch.split(':')[-1])
        if Epoch <= 400 and Epoch!=bf_epoch:
            MAE=float(MAE[4:])
            RMSE=float(RMSE[6:])
            pearson=float(pearson[9:])

            MAE_all.append(MAE)
            RMSE_all.append(RMSE)
            pearson_all.append(pearson)
            Epoch_all.append(Epoch)

            bf_epoch=Epoch
            # print(line)
        if Epoch > 400:
            break

Baseline_MAE_all=MAE_all.copy()
print(Baseline_MAE_all)

log_path = r"F:\pix2pix_FC\checkpoints\pix2pix_FC\log_MAE.txt"
f=open(log_path,'r',encoding='UTF-8')

MAE_all_2=[]
RMSE_all=[]
pearson_all=[]
Epoch_all=[]
bf_epoch=0

for line in f.readlines():
    line = line.strip()
    if 'Epoch:' in line:
        # Epoch:5,MAE:0.11857, RMSE:0.34286, pearson:0.64998
        Epoch, MAE, RMSE, pearson =line.split(',')
        Epoch=int(Epoch.split(':')[-1])
        if Epoch <= 400 and Epoch!=bf_epoch:
            MAE=float(MAE[4:])
            RMSE=float(RMSE[6:])
            pearson=float(pearson[9:])

            MAE_all_2.append(MAE)
            RMSE_all.append(RMSE)
            pearson_all.append(pearson)
            Epoch_all.append(Epoch)

            bf_epoch=Epoch
            # print(line)
        if Epoch > 400:
            break


results_list=[Baseline_MAE_all,MAE_all_2]
print(results_list[0])
# save directory
save_dir='results'
print(save_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# creat a figure，set the aspect ratio of figures and dpi
fig1 = plt.figure(figsize=(8, 5), dpi=200)
ax = plt.subplot(111)

"""
# in 'bo-', b is blue, o is O marker, - is solid line and so on
# color：r-red  g-green    b-blue  y-yellow  w-white  k-black  c-cyan  m-magenta
"""

markersize = 5
linewidth = 1.0
# with mark
line_style_list=['bo-','gv--','ms-.','ch:','rD-']
# without mark
line_style_list = ['b-', 'r-', 'c:', 'r-']
label_list=['Baseline','pix2pix with feature controller']
y_all=[]

# x=np.linspace(0,len(G_GAN_all)-1, len(G_GAN_all))
for idx,result in enumerate(results_list):
    y = results_list[idx]
    #assert len(x)==len(y)
    ax.plot(Epoch_all, y, line_style_list[idx], label=label_list[idx], markersize=markersize, linewidth=linewidth)


#para1：loc，repsent location
# best  upper right upper left    lower left  lower right right   center left center right     lower center
# upper center    center

# set legend property
plt.legend(loc=0, ncol=2, fontsize=13)
# set the label of x axis
plt.xlabel('Epochs', fontsize=15)
# set the label of y axis
plt.ylabel('MAE', fontsize=15)

# set the limitation of x
# plt.xlim([0, 300])
# set the limitation of y
plt.ylim([0.07, 0.20])

# save figure
plt.savefig(os.path.join(save_dir, log_path.split('/')[-1][:-4]+'_MAE_Ensemble.png'), dpi=200, bbox_inches='tight')
# show the figure
plt.show()
# close figure
plt.close()
