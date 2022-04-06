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


    score2 = m(img1, img2).item()
    score2 = (score2-0.8)*5
    return score2


# 扰动大小评分规则score_1，范围[0, 1]，越高越好
def getscore1(ori_img, adv_img):
    ori_img = np.array(ori_img, dtype=int)  # 图像数组，（height, weight, channels）
    adv_img = np.array(adv_img, dtype=int)
    dif = np.abs(adv_img - ori_img)

    """ #查看有多少个像素点在resize后变化特别大
    sum=0
    for i in range(dif.shape[0]):
        for j in range(dif.shape[1]):
            for k in range(dif.shape[2]):
                if dif[i][j][k]>20:
                    sum+=1
    all = dif.shape[0]*dif.shape[1]*dif.shape[2]
    print(all)
    print(sum)
    print(sum/all)
    """
    print(dif[:, :, 0].max() , dif[:, :, 1].max() , dif[:, :, 2].max())
   # dif = np.abs(np.clip((adv_img - ori_img), -20, 20) ) # 扰动限制在[-20, 20]的区间范围内
    score1 = 1 - (dif[:, :, 0].max() + dif[:, :, 1].max() + dif[:, :, 2].max()) / 60
    return score1


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
    cal_imgs('../source_data/images',
             '../result_data/images',
             './score.csv')