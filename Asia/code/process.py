import cv2
import os
import shutil
from align.align_dataset_mtcnn import main


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False



# 将需要攻击的图片合并成单级目录结构
def merge_1(src2, des2,des3=" "):
    folders = os.listdir(src2)
    determination = des2
    if not os.path.exists(determination):
        os.makedirs(determination)

    if des3 != " ":
        if not os.path.exists(des3):
            os.makedirs(des3)

    for folder in folders:
        if is_number(folder) :

            dir = src2 + '/' + str(folder)
            files = os.listdir(dir)
            for file in files:
                source = dir + '/' + str(file)
                deter2 = determination + '/' + str(file)
                shutil.copyfile(source, deter2)

                if file.split('.')[1] =='png':
                    jpg_img = deter2.replace('png', 'jpg')
                    os.rename(deter2, jpg_img)

                if des3 != " ":
                    deter3 = des3 + '/' + file
                    print(deter3)

                    shutil.copyfile(source, deter3)




input = '../source_data/images'
des1 = '../source_data/adv_data'  #用于生成对抗样本
output  = '../source_data/images112'
des2 = '../source_data/adv_data112'

merge_1(input, des1,des2)  # 计算相似度用
main(input, output)
merge_1(output, des2)

