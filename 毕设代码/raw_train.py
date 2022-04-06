"""
最好代码
使用最好攻击代码 攻击匹配图片 将loss设置为远离原图和目标图像 clip设置为3次 每次0.02
Ti使用0.08
不使用动量
"""
import pandas as pd
import json
import random
from guass import *
import os
from PIL import Image
import torchvision
import warnings
import torch.multiprocessing
from backbones.model_irse import IR_50, IR_101, IR_152, IR_SE_50, MobileFaceNet
from backbones.model_facenet import InceptionResnetV1
from calscore import getscore1, getscore2
import pytorch_msssim
import csv
from backbones import get_model
import time
import sys
from cal_attack import cal_attack_rate
from calscore import only_cal_imgs

warnings.filterwarnings("ignore")
to_torch_tensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

maxeps = 20.0
times = 4

def input_diversity(image, prob=0.2, low=112, high=128):
    if random.random() < prob:
        image = F.upsample(image, size=[low, low], mode='bicubic')  # bilinear
        return image
    rnd = random.randint(low, high)
    rescaled = F.upsample(image, size=[rnd, rnd], mode='bicubic')
    h_rem = high - rnd
    w_rem = high - rnd
    pad_top = random.randint(0, h_rem)
    pad_bottom = h_rem - pad_top
    pad_left = random.randint(0, w_rem)
    pad_right = w_rem - pad_left
    padded = F.pad(rescaled, [pad_top, pad_bottom, pad_left, pad_right], 'constant', 0)
    padded = F.upsample(padded, size=[low, low], mode='bicubic') #lukeda
    return padded


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


# TV loss
def tv_loss(input_t):
    temp1 = torch.cat((input_t[:, :, 1:, :], input_t[:, :, -1, :].unsqueeze(2)), 2)
    temp2 = torch.cat((input_t[:, :, :, 1:], input_t[:, :, :, -1].unsqueeze(3)), 3)
    temp = (input_t - temp1) ** 2 + (input_t - temp2) ** 2
    return temp.sum()


# 初始化噪声
def get_init_noise(device, input):
    noise = torch.Tensor(input.shape)
    # noise = torch.nn.init.xavier_normal(noise, gain=1)
    # noise = torch.nn.init.xavier_uniform(noise, gain=1)
    #noise = torch.nn.init.kaiming_uniform(noise)

    # noise = torch.nn.init.uniform_(noise, a=-8 / 255 * 2, b=8 / 255 * 2) #初始化噪声不能太大
    noise = torch.zeros_like(input) #不初始化噪声
    return noise.to(device)


# 返回人脸识别模型（主干网络）（输出512维的向量）
def get_all_model(model, param, device, proportion):
    m = model([112, 112])
    m.eval()
    m.to(device)

    m.load_state_dict(torch.load(param, map_location=device))
    model_dict = {'model': m, 'proportion': proportion}
    return model_dict


def get_other_model(weight, name, device, value, proportion):
    m = get_model(name, fp16=value)
    m.eval()
    m.to(device)

    m.load_state_dict(torch.load(weight, map_location=device))
    model_dict = {'model': m, 'proportion': proportion}
    return model_dict


def get_other_model2(name, device, proportion):
    m = InceptionResnetV1(pretrained='vggface2', device=device).eval()

    model_dict = {'model': m, 'proportion': proportion}
    return model_dict


def get_other_model3(param, device, proportion):
    m = MobileFaceNet(512)
    m.eval()
    m.to(device)

    m.load_state_dict(torch.load(param, map_location=device))
    model_dict = {'model': m, 'proportion': proportion}
    return model_dict

def get_other_model4( device, proportion):
    m = insightface.iresnet100(pretrained=True)
    m.eval()
    m.to(device)

    model_dict = {'model': m, 'proportion': proportion}
    return model_dict

# 返回模型池
def get_model_pool(device, which):
    model_pool = []
    if which == 1:
        model_pool.append(get_all_model(IR_50, '../user_data/models/backbone_ir50_ms1m_epoch120.pth', device,1))  # 单模 155.708878，https://github.com/ZhaoJ9014/face.evoLVe.PyTorch?spm=5176.21852664.0.0.7a6570e7wfEMmH&file=face.evoLVe.PyTorch#Model-Zoo

    elif which == 2:
        model_pool.append(get_all_model(IR_152, '../user_data/models/Backbone_IR_152_MS1M_Epoch_112.pth', device,1))  # 单模 159.544693,https://github.com/ZhaoJ9014/face.evoLVe.PyTorch?spm=5176.21852664.0.0.7a6570e7wfEMmH&file=face.evoLVe.PyTorch#Model-Zoo
    elif which == 3:
        model_pool.append(get_other_model3('../user_data/models/model_mobilefacenet.pth', device,
                                           1))  # 单模192.300568，https://github.com/TreB1eN/InsightFace_Pytorch

        #model_pool.append(get_other_model('../user_data/models/ms1mv3_arcface_r50_fp16/backbone.pth', 'r50',device,True,1)) #94.943787
    elif which == 4:
        model_pool.append(get_all_model(IR_SE_50, '../user_data/models/model_ir_se50.pth', device,
                                        1))  # 250.737747  https://github.com/TreB1eN/InsightFace_Pytorch

    elif which == 5:
        model_pool.append(get_all_model(IR_50, '../user_data/models/backbone_ir50_asia.pth', device,1))  # 131.105804 Aisa数据集，https://github.com/ZhaoJ9014/face.evoLVe.PyTorch?spm=5176.21852664.0.0.7a6570e7wfEMmH&file=face.evoLVe.PyTorch#Model-Zoo

    #model_pool.append(get_all_model(IR_50, '../user_data/models/backbone_ir50_asia.pth', device,1))  # 131.105804 Aisa数据集，https://github.com/ZhaoJ9014/face.evoLVe.PyTorch?spm=5176.21852664.0.0.7a6570e7wfEMmH&file=face.evoLVe.PyTorch#Model-Zoo

    # 以下为同一个文件
    # model_pool.append(get_all_model(IR_50, '../user_data/models/backbone_ir50_ms1m_epoch120.pth', device, 1))  #单模 155.708878，https://github.com/ZhaoJ9014/face.evoLVe.PyTorch?spm=5176.21852664.0.0.7a6570e7wfEMmH&file=face.evoLVe.PyTorch#Model-Zoo
    #model_pool.append(get_all_model(IR_50, '../user_data/models/Backbone_IR_50_LFW.pth', device, 1)) #单模 225.52
    #model_pool.append(
    #    get_all_model(IR_101, '../user_data/models/Backbone_IR_101_Batch_108320.pth', device, 1))  # 单模244.621078
    #model_pool.append(get_all_model(IR_152, '../user_data/models/Backbone_IR_152_MS1M_Epoch_112.pth', device, 1)) #单模 159.544693,https://github.com/ZhaoJ9014/face.evoLVe.PyTorch?spm=5176.21852664.0.0.7a6570e7wfEMmH&file=face.evoLVe.PyTorch#Model-Zoo
    #model_pool.append(get_all_model(IR_50, '../user_data/models/Backbone_IR_50_LFW_ADV_TRAIN.pth', device,
    #                                1))  # 居然有对抗训练的模型，得试试#单模227.112106，对抗效果一般般
    # model_pool.append(get_all_model(IR_50, '../user_data/models/backbone_ir50_asia.pth', device,1))  # 131.105804 Aisa数据集，https://github.com/ZhaoJ9014/face.evoLVe.PyTorch?spm=5176.21852664.0.0.7a6570e7wfEMmH&file=face.evoLVe.PyTorch#Model-Zoo
    #model_pool.append(get_all_model(IR_SE_50, '../user_data/models/model_ir_se50.pth', device,
    #                                1))  # 250.737747  https://github.com/TreB1eN/InsightFace_Pytorch

    # model_pool.append(get_other_model('../user_data/models/ms1mv3_arcface_r100_fp16/backbone.pth', 'r100',device,True,1)) #94.943787
    # model_pool.append(get_other_model('../user_data/models/ms1mv3_arcface_r2060/backbone.pth', 'r2060', device, False, 1))#88.056427
    # model_pool.append(get_other_model('../user_data/models/glint360k_cosface_r100_fp16_0.1/backbone.pth', 'r100', device, True,1))  #94.654289
    #elif which == 6:
    #    model_pool.append(get_other_model2('vggface2', device, 1))  # 287.170624 https://github.com/timesler/facenet-pytorch
    #model_pool.append(
    #    get_other_model2('casia-webface', device, 1))  # 290.522003 https://github.com/timesler/facenet-pytorch

    # 以下为另一个文件


    return model_pool


# 设置各个模型权重
def normal_model_proportion(model_pool):
    sum1 = 0
    for model_dict in model_pool:
        sum1 += model_dict['proportion']
    for model_dict in model_pool:
        model_dict['proportion'] /= sum1
    return model_pool




# 多步迭代, 调用单步迭代函数
def noise_iter(model_pool, origin_img, target_img, gaussian_blur, device, break_threshold, flag,origin_path):

    target_img_pool = F.upsample(target_img, size=[112, 112], mode='bicubic')
    origin_img_pool = F.upsample(origin_img, size=[112, 112], mode='bicubic')


    for model_dict in model_pool:
        model = model_dict['model']
        proportion = model_dict['proportion']
        v2 = l2_norm(model(origin_img_pool)).detach_()  # 原图向量
        v2_1 = l2_norm(model(target_img_pool)).detach_()  # 目标图片向量

        # 用的余弦相似度，越大越相似
        tmp1 = (v2 * v2_1).sum()  # 对抗样本 与 匹配图像 的向量的内积


    yield tmp1


# 计算单个对抗样本
def cal_adv(origin_path, target_path, model_pool, gaussian_blur, device, break_threshold):
    origin_img = Image.open(origin_path).convert('RGB')
    # origin_img = origin_img.resize((112, 112), Image.NEAREST)
    origin_img = to_torch_tensor(origin_img)
    origin_img = origin_img.unsqueeze_(0).to(device)

    target_img = Image.open(target_path).convert('RGB')
    #target_img = target_img.resize((112, 112), Image.NEAREST)
    target_img = to_torch_tensor(target_img)
    target_img = target_img.unsqueeze_(0).to(device)


    flag = 0  # 取times关掉clip，取0开启clip
    generator = noise_iter(model_pool, origin_img, target_img,gaussian_blur, device, break_threshold, flag,origin_path)


    tmp1 = next(generator)


    return tmp1


def attack(face_path, origin_name, tar_path, model_pool, gaussian_blur, device, break_threshold, tatget):
    tmp1 = cal_adv(os.path.join(face_path, origin_name),
                              tar_path,
                              model_pool,
                              gaussian_blur,
                              device,
                              break_threshold)
    # print(noise.shape)
    tmp1 = tmp1.cpu().numpy()
    # print(noise.shape)

    #tmp1

    f = open('iter_num.txt', 'a')
    f.write(os.path.join(face_path, origin_name) + ';' + tar_path + ';' + str(tmp1) + '\n')
    f.close()

    return tmp1


# 单进程运行
def one_process_run(people_list, model_pool, device, face_path, tatget):

    #if os.path.exists('iter_num.txt'):
    #    os.remove('iter_num.txt')

    #likelihood = json.load(open("maxlikelihood_images.json"))
    # gaussian_blur = get_gaussian_blur(kernel_size=3, device=device)  # 高斯滤波：消除高斯噪声，在图像处理的降噪、平滑中应用较多
    kernel = gkern(3, 3).astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 1)
    gaussian_blur = torch.Tensor(stack_kernel).to(device)

    #print(people_list)
    #print(len(people_list))

    for origin_name in people_list:

        if origin_name.split('_')[1] == '0.jpg':
            tar_name = origin_name.split('_')[0] + '_0.jpg'  # 自己类的第一张图片
        else:
            tar_name = origin_name.split('_')[0] + '_0.jpg'  # 自己类的第一张图片


        tar_path = os.path.join("../source_data/adv_data112", tar_name)  # 自己类的第一张图片

        # 生成当前图像的对抗样本
        score3 = 1
        break_threshold = -0.35

        #重启获得最优解，我发现有的插值后效果会变差，有的会变好
        #while score3 > 0.001:  # 保证resize后攻击相似度小于0
        score3 = attack(face_path, origin_name, tar_path, model_pool, gaussian_blur, device, break_threshold, tatget)
            #break_threshold -= 0.02

            #score3 = -1 #只进行一次攻击，不通过线下得到的分数进行干预


# Utilize one GPU to generate an adversarial sample with four processing
def one_device_run(p_pool, people_list, device, face_path,which):
    four_process = True
    model_pool = get_model_pool(device,which)
    model_pool = normal_model_proportion(model_pool)
    print('----model load over----')
    if four_process:
        for model_dict in model_pool:
            model_dict['model'].share_memory()

        """
        p_pool.apply_async(one_process_run,
                           args=(people_list[:len(people_list) // 2], model_pool, device, face_path, 0))
        p_pool.apply_async(one_process_run, args=(
            people_list[len(people_list) // 2:], model_pool, device, face_path, 0))

        """
        p_pool.apply_async(one_process_run,
                           args=(people_list[:len(people_list) // 8], model_pool, device, face_path, 0))
        p_pool.apply_async(one_process_run, args=(
            people_list[len(people_list) // 8:len(people_list) // 8 * 2], model_pool, device, face_path, 0))
        p_pool.apply_async(one_process_run, args=(
            people_list[len(people_list) // 8 * 2: len(people_list) // 8 * 3], model_pool, device, face_path, 0))
        p_pool.apply_async(one_process_run, args=(
            people_list[len(people_list) // 8 * 3:len(people_list) // 8 * 4], model_pool, device, face_path, 0))
        p_pool.apply_async(one_process_run, args=(
            people_list[len(people_list) // 8 * 4:len(people_list) // 8 * 5], model_pool, device, face_path, 0))
        p_pool.apply_async(one_process_run, args=(
            people_list[len(people_list) // 8 * 5:len(people_list) // 8 * 6], model_pool, device, face_path, 0))
        p_pool.apply_async(one_process_run, args=(
            people_list[len(people_list) // 8 * 6: len(people_list) // 8 * 7], model_pool, device, face_path, 0))
        p_pool.apply_async(one_process_run,
                           args=(people_list[len(people_list) // 8 * 7:], model_pool, device, face_path, 0))

    else:
        p_pool.apply_async(one_process_run, args=(people_list, model_pool, device))


def mutil_device_run(which):
    if os.path.exists('iter_num.txt'):
        os.remove('iter_num.txt')

    face_path = "../result_data/vgg_DIV_MI_16_adv_images"
    #face_path = "../source_data/adv_data112"

    # device_list 是显卡列表， task_num_list 是每个设备生成对抗样本数量
    device_list = [ 'cuda:0', 'cuda:1']
    task_num_list = [ 500, 500]

    # device_list = ['cuda:0', 'cuda:3']
    # task_num_list = [250, 250]
    if len(task_num_list) != len(device_list):
        raise 'task_num_list is not same as device_list!'
    if np.array(task_num_list).sum() != 1000:
        raise 'imgs num is not 500!'

    faces = os.listdir(face_path)

    # 使用多进程实现多显卡同时运行， 也可以实现单显卡的加速
    start_index = 0
    p_pool = torch.multiprocessing.Pool()
    for i, device in enumerate(device_list):
        one_device_run(p_pool, faces[start_index:start_index + task_num_list[i]], device, face_path,which)
        start_index += task_num_list[i]
    p_pool.close()
    p_pool.join()


# 单GPU四进程
def eight_process_run():
    face_path = "../source_data/adv_data112"

    # device_list 是显卡列表， task_num_list 是每个设备生成对抗样本数量
    # device_list = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']
    # task_num_list = [125, 125, 125, 125]

    device_list = ['cuda:0']
    task_num_list = [1000]
    if len(task_num_list) != len(device_list):
        raise 'task_num_list is not same as device_list!'
    if np.array(task_num_list).sum() != 1000:
        raise 'imgs num is not 1000!'

    faces = os.listdir(face_path)

    e1 = time.time()
    # 使用多进程实现多显卡同时运行， 也可以实现单显卡的加速
    start_index = 0
    p_pool = torch.multiprocessing.Pool()
    for i, device in enumerate(device_list):
        one_device_run(p_pool, faces[start_index:start_index + task_num_list[i]], device, face_path)
        start_index += task_num_list[i]
    p_pool.close()
    p_pool.join()
    e2 = time.time()
    print ("并行执行时间：", int(e2 - e1))


# 单进程运行
def one_process_run_solo(which,face_path):
    if os.path.exists('hard.txt'):
        os.remove('hard.txt')
    if os.path.exists('iter_num.txt'):
        os.remove('iter_num.txt')

    device = torch.device('cuda:2')
    model_pool = normal_model_proportion(get_model_pool(device,which))
    people_list = os.listdir(face_path)
    e1 = time.time()
    one_process_run(people_list, model_pool, device, face_path,0)
    e2 = time.time()
    print ("solo执行时间: ", int(e2 - e1))

if __name__ == '__main__':
    #one_process_run_solo()
    #which = int(sys.argv[1])
    #print(which)
    #mutil_device_run(which)
    #one_process_run_solo(which)

    face_path = sys.argv[1]#"../source_data/adv_data112"#"../result_data/vgg_DIV_MI_16_adv_images"  # 攻击图像的地址
    target_path = sys.argv[2]#"../save_data/results.csv"
    df_new = pd.DataFrame()
    df_new['name'] = ['m1', 'm2', 'm3', 'm4', 'm5']
    df_new['attack_rate'] = -1
    df_new['image_num'] = -1
    df_new['cos_mean'] = -1
    df_new['SSIM'] = -1
    df_new['L无穷3_score'] = -1
    df_new['L无穷_score'] = -1
    df_new['L2_score'] = -1
    score1, score2, score3, score4 = only_cal_imgs(face_path, ori_dir="../source_data/adv_data112")
    df_new.loc[0, 'L无穷3_score'] = score1
    df_new.loc[0, 'SSIM'] = score2
    df_new.loc[0, 'L无穷_score'] = score3
    df_new.loc[0, 'L2_score'] = score4

    for i in range(5):
        which = i+1
        print(which)
        one_process_run_solo(which,face_path)
        attack_rate, num , cos_mean = cal_attack_rate()
        df_new.loc[i, 'attack_rate'] = attack_rate
        df_new.loc[i, 'image_num'] = num
        df_new.loc[i, 'cos_mean'] = cos_mean



    if os.path.exists(target_path):
        save_df = pd.read_csv(target_path)
        save_df = pd.concat([save_df, df_new])
    else:
        save_df = df_new

    save_df.to_csv(target_path,index=None)
    exit()
