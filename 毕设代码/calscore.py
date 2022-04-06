from PIL import Image
import numpy as np
import torch
import pytorch_msssim
import csv
import os
import imageio
# 图像质量评分score_2，范围[0, 1]，越高越好
def getscore2(ori_img, adv_img):
    m = pytorch_msssim.MSSSIM()

    ori_img = np.array(ori_img, dtype=float)  # 图像数组，（height, weight, channels）
    adv_img = np.array(adv_img, dtype=float)
    img1 = torch.from_numpy(ori_img.transpose((2, 0, 1)) / 255*2-1).float().unsqueeze(0)   #放缩到-1到1输进去其实问题也不大
    img2 = torch.from_numpy(adv_img.transpose((2, 0, 1)) / 255*2-1).float().unsqueeze(0)


    score2 = float(m(img1, img2).item())
    if np.isnan(score2):
        return 0.8
    else: return score2


# 三通道L无穷
def getscore1(ori_img, adv_img):
    ori_img = np.array(ori_img, dtype=int)  # 图像数组，（height, weight, channels）
    adv_img = np.array(adv_img, dtype=int)
    dif = np.abs(adv_img - ori_img)
    score1 =  (dif[:, :, 0].max()+ dif[:, :, 1].max()+dif[:, :, 2].max())/3   #1 - (dif[:, :, 0].max() + dif[:, :, 1].max() + dif[:, :, 2].max()) / 60
    return score1


# L无穷
def getscore3(ori_img, adv_img):
    ori_img = np.array(ori_img, dtype=int)  # 图像数组，（height, weight, channels）
    adv_img = np.array(adv_img, dtype=int)
    dif = np.abs(adv_img - ori_img)
    score3 =  dif.max()   #1 - (dif[:, :, 0].max() + dif[:, :, 1].max() + dif[:, :, 2].max()) / 60
    return score3

# L2
def getscore4(ori_img, adv_img):
    ori_img = np.array(ori_img, dtype=int)  # 图像数组，（height, weight, channels）
    adv_img = np.array(adv_img, dtype=int)
    dif = np.abs(adv_img - ori_img)
    score4 =  np.sqrt(np.sum((dif)**2))   #1 - (dif[:, :, 0].max() + dif[:, :, 1].max() + dif[:, :, 2].max()) / 60
    return score4


def only_cal_imgs(adv_dir,ori_dir="../source_data/adv_data112"):
    score1s = []
    score2s = []
    score3s = []
    score4s = []

    imgs = os.listdir(adv_dir)
    for img in imgs:
        ori_img = Image.open(os.path.join(ori_dir, img)).convert('RGB')
        adv_img = Image.open(os.path.join(adv_dir, img)).convert('RGB')
        ori_img = np.array(ori_img, dtype=int)  # 图像数组，（height, weight, channels）
        adv_img = np.array(adv_img, dtype=int)

        score1 = getscore1(ori_img, adv_img)
        score1s.append(score1)

        score2 = getscore2(ori_img, adv_img)
        score2s.append(score2)


        score3 = getscore3(ori_img, adv_img)
        score3s.append(score3)

        score4 = getscore4(ori_img, adv_img)
        score4s.append(score4)

    print(len(score2s))
    #print(score2s)
    print(np.mean(np.array(score1s)))
    print(np.mean(np.array(score2s)))
    print(np.mean(np.array(score3s)))
    print(np.mean(np.array(score4s)))

    return np.mean(np.array(score1s)),np.mean(np.array(score2s)),np.mean(np.array(score3s)),np.mean(np.array(score4s))

def cal_imgs(ori_dir,adv_dir,output_path):
    with open(output_path,'w',newline='') as file:
        writer=csv.writer(file)
        writer.writerow(['image','score1','score2'])
        labels=os.listdir(adv_dir)

        score1s = []
        score2s = []
        for label in labels:
            if label =='.DS_Store':
                continue
            imgs=os.listdir(os.path.join(adv_dir,label))
            for img in imgs:
                ori_img = Image.open(os.path.join(ori_dir,label,img)).convert('RGB')
                adv_img= Image.open(os.path.join(adv_dir,label,img)).convert('RGB')
                ori_img = np.array(ori_img, dtype=int)  # 图像数组，（height, weight, channels）
                adv_img = np.array(adv_img, dtype=int)

                score1=getscore1(ori_img,adv_img)
                score1s.append(score1)

                score2=getscore2(ori_img,adv_img)
                score2s.append(score2)
                writer.writerow([label+'/'+img,score1,score2])
                print(label+'/'+img+' score1: '+str(score1)+' score2: '+str(score2))

        print(len(score1s))
        print(np.sum(np.array(score1s)))
        print(np.sum(np.array(score2s)))

if __name__ == '__main__':
    only_cal_imgs('../result_data/casia_DIV_MI_16_adv_images')