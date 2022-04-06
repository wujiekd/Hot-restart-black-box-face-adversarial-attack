"""
最好代码
使用最好攻击代码 攻击匹配图片 将loss设置为远离原图和目标图像 clip设置为3次 每次0.02
Ti使用0.08
不使用动量
"""

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
warnings.filterwarnings("ignore")

to_torch_tensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


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
    #noise = torch.Tensor(input.shape)
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
def get_model_pool(device):
    model_pool = []

    #model_pool.append(get_other_model2('vggface2', device, 1))  # 287.170624 https://github.com/timesler/facenet-pytorch
    model_pool.append(
        get_other_model2('casia-webface', device, 1))  # 290.522003 https://github.com/timesler/facenet-pytorch


    # 以下为同一个文件
    # model_pool.append(get_all_model(IR_50, '../user_data/models/backbone_ir50_ms1m_epoch120.pth', device, 1))  #单模 155.708878，https://github.com/ZhaoJ9014/face.evoLVe.PyTorch?spm=5176.21852664.0.0.7a6570e7wfEMmH&file=face.evoLVe.PyTorch#Model-Zoo

    # model_pool.append(get_other_model('../user_data/models/ms1mv3_arcface_r2060/backbone.pth', 'r2060', device, False, 1))#88.056427
    # model_pool.append(get_other_model('../user_data/models/glint360k_cosface_r100_fp16_0.1/backbone.pth', 'r100', device, True,1))  #94.654289

    #model_pool.append(get_other_model2('vggface2', device, 1))  # 287.170624 https://github.com/timesler/facenet-pytorch
    #model_pool.append(
    #    get_other_model2('casia-webface', device, 1))  # 290.522003 https://github.com/timesler/facenet-pytorch

    # 以下为另一个文件
    #model_pool.append(get_other_model3('../user_data/models/model_mobilefacenet.pth', device,
     #                                  1))  # 单模192.300568，https://github.com/TreB1eN/InsightFace_Pytorch


    return model_pool


# 设置各个模型权重
def normal_model_proportion(model_pool):
    sum1 = 0
    for model_dict in model_pool:
        sum1 += model_dict['proportion']
    for model_dict in model_pool:
        model_dict['proportion'] /= sum1
    return model_pool


# 单步迭代
def iter_step(tmp_noise, origin_img, target_img, gaussian_blur, model_pool, index, loss1_v, momentum, lr, eps1, eps2,
              eps3):
    tmp_noise.requires_grad = True
    noise = tmp_noise  # noise = gaussian_blur(tmp_noise)   #高斯平滑
    loss1 = 0
    score1 = 0
    score2 = 0

    target_img_pool = F.upsample(target_img, size=[112, 112], mode='bicubic')
    origin_img_pool = F.upsample(origin_img, size=[112, 112], mode='bicubic')
    adv_image = origin_img + noise
    #adv_image_pool = input_diversity(adv_image)  # 开启输入多样性
    for model_dict in model_pool:
        model = model_dict['model']
        proportion = model_dict['proportion']
        v1 = l2_norm(model(adv_image))  # 对抗样本向量
        v2_1 = l2_norm(model(origin_img_pool)).detach_()  # 原图向量
        #v2_2 = l2_norm(model(target_img_pool)).detach_()  # 目标图片向量

        # 用的余弦相似度，越大越相似
        tmp1 = (v1 * v2_1).sum()  # 对抗样本 与 原图 的向量的内积
        #tmp2 = (v1 * v2_2).sum()  # 对抗样本 与 目标图片 的向量的内积



        loss1 += tmp1 * proportion  # (tmp1 - tmp2) * proportion  # 只采用非定向，不使用定向，集成模型
        #score1 += tmp1.item() * proportion
        score1 += tmp1.item() * proportion

    #m = pytorch_msssim.MSSSIM()

    #msssim = m(origin_img, adv_image).item()  # 正则化项,越大越好，可以优化,但好像不加它效果更好
    #loss3 = (msssim - 0.8) * 5

    # print(loss1)
    # print(loss3)
    all_loss = loss1 #- loss3

    all_loss.backward(retain_graph=True)  # 反向传播更新噪声

    """78.368706
    # 对梯度进行归一化，这个lr得用小的1/127.5，因为类似于符号函数
    #print(torch.max(tmp_noise.grad))
    grad_norms = tmp_noise.grad.view(1, -1).norm(p=float('inf'), dim=1) #算L无穷
    #print(grad_norms)
    tmp_noise.grad.div_(grad_norms.view(-1, 1, 1, 1))

    #写个动量I-fgsm，这个lr得用大的，对梯度归一化后，数值其实蛮小的，lr用1
    #print(torch.max(tmp_noise.grad))
    grad_norms = tmp_noise.grad.view(1, -1).norm(p=1, dim=1)    #算L1距离，还要取均值的
    #print(grad_norms)
    tmp_noise.grad.div_(grad_norms.view(-1, 1, 1, 1))

    loss1_v = tmp_noise.grad.detach() + loss1_v * momentum  # 取消动量项，利用loss1更新扰动
    """

    now_loss1_v = tmp_noise.grad.detach()

    #loss1_v_gauss = F.conv2d(now_loss1_v, gaussian_blur, padding=(3 - 1) // 2, groups=3)  # 动量可以试着加在平滑后面

    #L1_grad_norms = now_loss1_v.view(1, -1).norm(p=1, dim=1)  # 使用L1进行正则化
    #L1_norm_loss1_v = now_loss1_v.div(L1_grad_norms.view(-1, 1, 1, 1))

    #if (L1_grad_norms == 0).any():  # avoid nan or inf if gradient is 0
       # L1_norm_loss1_v = torch.randn_like(L1_grad_norms)

    #loss1_v = L1_norm_loss1_v + loss1_v * momentum  # 使用动量项，利用loss1更新扰动
    # print(torch.max(tmp_noise.grad))

    #grad_norms = loss1_v.view(1, -1).norm(p=float('inf'), dim=1)  # 使用L无穷进行正则化
    #norm_loss1_v = loss1_v.div(grad_norms.view(-1, 1, 1, 1))


    tmp_noise = tmp_noise.detach() - lr * torch.sign(now_loss1_v)  # 不用正则化

    tmp_noise[0][0] = tmp_noise[0][0].clamp_(-eps1, eps1)  # clip
    tmp_noise[0][1] = tmp_noise[0][1].clamp_(-eps2, eps2)  # clip
    tmp_noise[0][2] = tmp_noise[0][2].clamp_(-eps3, eps3)  # clip

    tmp_noise = (tmp_noise + origin_img).clamp_(-1, 1) - origin_img

    """
    print('————————————————————————————————————————————————')
    print(eps)
    print(np.quantile(tmp_noise.cpu()[0][0].cpu(), 1.0))
    print(np.quantile(tmp_noise.cpu()[0][0].cpu(), 0.95))
    print(np.quantile(tmp_noise.cpu()[0][0].cpu(), 0.9))
    print(np.quantile(tmp_noise.cpu()[0][0].cpu(), 0.8))
    print(np.quantile(tmp_noise.cpu()[0][1].cpu(), 0.8))
    print(np.quantile(tmp_noise.cpu()[0][2].cpu(), 0.8))
    """
    return tmp_noise, score1, score2, loss1_v, eps1, eps2, eps3


# 多步迭代, 调用单步迭代函数
def noise_iter(model_pool, origin_img, target_img, gaussian_blur, device, break_threshold, flag,origin_path,maxeps):
    learning_rate = maxeps / 255 * 2   # 可以调整,步长为1，因为图片放缩了，步长也要放缩

    momentum = 0.9  # 关闭动量
    tmp_noise = get_init_noise(device, origin_img)  # 初始化噪声
    tmp_noise_save = tmp_noise.clone()  # 存储上一次进入判断的噪声
    index = 0
    loss1_v = 0
    eps1 = maxeps / 255 * 2  # 起始最大eps
    eps2 = maxeps / 255 * 2
    eps3 = maxeps / 255 * 2


    while True:
        last_tmp_noise = tmp_noise.clone()  #score1计算的是上一次noise的分数
        index += 1

        tmp_noise, score1, score2, loss1_v, eps1, eps2, eps3 = iter_step(last_tmp_noise, origin_img, target_img, gaussian_blur, model_pool,
                                                       index, loss1_v, momentum, learning_rate, eps1, eps2, eps3)

        yield tmp_noise,last_tmp_noise, score1, score2, flag, tmp_noise_save, index


# 计算单个对抗样本
def cal_adv(origin_path, target_path, model_pool, gaussian_blur, device, break_threshold,maxeps):
    origin_img = Image.open(origin_path).convert('RGB')
    # origin_img = origin_img.resize((112, 112), Image.NEAREST)
    origin_img = to_torch_tensor(origin_img)
    origin_img = origin_img.unsqueeze_(0).to(device)

    target_img = Image.open(target_path).convert('RGB')
    #target_img = target_img.resize((112, 112), Image.NEAREST)
    target_img = to_torch_tensor(target_img)
    target_img = target_img.unsqueeze_(0).to(device)


    flag = 0  # 取times关掉clip，取0开启clip
    max_iterations = 1
    generator = noise_iter(model_pool, origin_img, target_img,gaussian_blur, device, break_threshold, flag,origin_path,maxeps)
    scores = 0
    i = 0

    while True:
        tmp_noise,last_tmp_noise, score1, score2, flag, tmp_noise_save, index = next(generator)

        i += 1

        if i >= max_iterations:  #总迭代次数不能超过300代，不然使用上次的攻击代数

           break


    return tmp_noise  , i


def attack(face_path, origin_name, tar_path, model_pool, gaussian_blur, device, break_threshold, tatget,maxeps):
    noise, iter_num = cal_adv(os.path.join(face_path, origin_name),
                              tar_path,
                              model_pool,
                              gaussian_blur,
                              device,
                              break_threshold,
                              maxeps)
    # print(noise.shape)
    noise = torch.round(noise * 127.5)[0].cpu().numpy()
    # print(noise.shape)
    noise = noise.swapaxes(0, 1).swapaxes(1, 2)
    noise = noise.clip(-maxeps, maxeps)
    print(noise)

    origin_img = Image.open(os.path.join(face_path, origin_name)).convert('RGB')
    # raw_img = np.array(origin_img, dtype=float)
    # h = np.array(origin_img).shape[0]
    # w = np.array(origin_img).shape[1]


    # origin_img = origin_img.resize((112, 112), Image.NEAREST)  # 添加噪声
    origin_img = np.array(origin_img, dtype=float)
    numpy_adv_sample = (origin_img + noise).clip(0, 255)
    adv_sample = Image.fromarray(np.uint8(numpy_adv_sample))


    # 保存图像
    output_dir = '../result_data/adv_images' + str(int(maxeps)) +'/'
    if os.path.exists(output_dir) is False:
        os.mkdir(output_dir)
    jpg_img = output_dir + origin_name
    png_img = jpg_img.replace('jpg', 'png')
    adv_sample.save(png_img)
    os.rename(png_img, jpg_img)  # 这样可以达到无损～

    # 读取图片进行评分
    # score评价
    ori_img = Image.open(os.path.join(face_path, origin_name)).convert('RGB')
    adv_img = Image.open(jpg_img).convert('RGB')
    target_img = Image.open(tar_path).convert('RGB')
    # target_img = target_img.resize((112, 112), Image.NEAREST)
    target_img = to_torch_tensor(target_img)
    target_img = target_img.unsqueeze_(0).to(device)

    score1 = getscore1(ori_img, adv_img)
    score2 = getscore2(ori_img, adv_img)
    print('%s noise score1 is %.4f' % (origin_name, score1))
    print('%s noise score2 is %.4f' % (origin_name, score2))

    # ori_img = ori_img.resize((112, 112), Image.NEAREST)  # 添加噪声
    # adv_img = adv_img.resize((112, 112), Image.NEAREST)  # 添加噪声

    # 读取图片进行评分
    origin_img = to_torch_tensor(ori_img)
    origin_img = origin_img.unsqueeze_(0).to(device)
    adv_img = to_torch_tensor(adv_img)
    adv_img = adv_img.unsqueeze_(0).to(device)

    origin_img = F.upsample(origin_img, size=[112, 112], mode='bicubic')
    adv_img = F.upsample(adv_img, size=[112, 112], mode='bicubic')
    target_img = F.upsample(target_img, size=[112, 112], mode='bicubic')

    score3_1 = 0
    score3_2 = 0
    for model_dict in model_pool:
        model = model_dict['model']
        proportion = model_dict['proportion']
        v1 = l2_norm(model(adv_img))  # 对抗样本向量
        v2_1 = l2_norm(model(origin_img)).detach_()  # 原图向量
        v2_2 = l2_norm(model(target_img)).detach_()
        # 用的余弦相似度，越大越相似
        tmp1 = (v1 * v2_1).sum()  # 对抗样本 与 原图 的向量的内积
        tmp2 = (v1 * v2_2).sum()
        score3_1 += tmp1.item() * proportion
        score3_2 += tmp2.item() * proportion
    print('%s noise score3_1(me) is %.4f' % (origin_name, score3_1))
    print('%s noise score3_2(tar) is %.4f' % (origin_name, score3_2))

    f = open('iter_num.txt', 'a')
    f.write(os.path.join(face_path, origin_name) + ';' + tar_path + ';' + str(score1) + ';' + str(score2) + ';' + str(
        score3_1) + ';' + str(
        score3_2) + '\n')
    f.close()

    return (score3_1+score3_2)/2


# 单进程运行
def one_process_run(people_list, model_pool, device, face_path, tatget,maxeps):
    if os.path.exists('hard.txt'):
        os.remove('hard.txt')
    if os.path.exists('iter_num.txt'):
        os.remove('iter_num.txt')

    #likelihood = json.load(open("maxlikelihood_images.json"))
    # gaussian_blur = get_gaussian_blur(kernel_size=3, device=device)  # 高斯滤波：消除高斯噪声，在图像处理的降噪、平滑中应用较多
    kernel = gkern(3, 3).astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 1)
    gaussian_blur = torch.Tensor(stack_kernel).to(device)

    print(people_list)
    print(len(people_list))

    for origin_name in people_list:

        if origin_name.split('_')[1] == '0.jpg':
            tar_name = origin_name.split('_')[0] + '_0.jpg'  # 自己类的第二张图片
        else:
            tar_name = origin_name.split('_')[0] + '_0.jpg'  # 自己类的第一张图片


        tar_path = os.path.join("../source_data/adv_data112", tar_name)  # 自己类的第一张图片

        # 生成当前图像的对抗样本
        score3 = 1
        break_threshold = -0.35

        #重启获得最优解，我发现有的插值后效果会变差，有的会变好
        #while score3 > 0.001:  # 保证resize后攻击相似度小于0
        score3 = attack(face_path, origin_name, tar_path, model_pool, gaussian_blur, device, break_threshold, tatget,maxeps)
            #break_threshold -= 0.02

            #score3 = -1 #只进行一次攻击，不通过线下得到的分数进行干预


# Utilize one GPU to generate an adversarial sample with four processing
def one_device_run(p_pool, people_list, device, face_path,maxeps):
    four_process = True
    model_pool = get_model_pool(device)
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
                           args=(people_list[:len(people_list) // 8], model_pool, device, face_path, 0,maxeps))
        p_pool.apply_async(one_process_run, args=(
            people_list[len(people_list) // 8:len(people_list) // 8 * 2], model_pool, device, face_path, 0,maxeps))
        p_pool.apply_async(one_process_run, args=(
            people_list[len(people_list) // 8 * 2: len(people_list) // 8 * 3], model_pool, device, face_path, 0,maxeps))
        p_pool.apply_async(one_process_run, args=(
            people_list[len(people_list) // 8 * 3:len(people_list) // 8 * 4], model_pool, device, face_path, 0,maxeps))
        p_pool.apply_async(one_process_run, args=(
            people_list[len(people_list) // 8 * 4:len(people_list) // 8 * 5], model_pool, device, face_path, 0,maxeps))
        p_pool.apply_async(one_process_run, args=(
            people_list[len(people_list) // 8 * 5:len(people_list) // 8 * 6], model_pool, device, face_path, 0,maxeps))
        p_pool.apply_async(one_process_run, args=(
            people_list[len(people_list) // 8 * 6: len(people_list) // 8 * 7], model_pool, device, face_path, 0,maxeps))
        p_pool.apply_async(one_process_run,
                           args=(people_list[len(people_list) // 8 * 7:], model_pool, device, face_path, 0,maxeps))

    else:
        p_pool.apply_async(one_process_run, args=(people_list, model_pool, device,maxeps))


def mutil_device_run(maxeps):
    face_path = "../source_data/adv_data112"     # 指定图片输入路径

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
        one_device_run(p_pool, faces[start_index:start_index + task_num_list[i]], device, face_path,maxeps)
        start_index += task_num_list[i]
    p_pool.close()
    p_pool.join()


# 单GPU四进程
def eight_process_run(maxeps):
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
        one_device_run(p_pool, faces[start_index:start_index + task_num_list[i]], device, face_path,maxeps)
        start_index += task_num_list[i]
    p_pool.close()
    p_pool.join()
    e2 = time.time()
    print ("并行执行时间：", int(e2 - e1))


# 单进程运行
def one_process_run_solo():
    face_path = "../source_data/adv_data112"  # 攻击图像的地址
    if os.path.exists('hard.txt'):
        os.remove('hard.txt')
    if os.path.exists('iter_num.txt'):
        os.remove('iter_num.txt')

    device = torch.device('cuda:0')
    model_pool = normal_model_proportion(get_model_pool(device))
    people_list = os.listdir(face_path)
    e1 = time.time()
    one_process_run(people_list, model_pool, device, face_path,0)
    e2 = time.time()
    print ("solo执行时间：", int(e2 - e1))

if __name__ == '__main__':
    #one_process_run_solo()
    maxeps = int(sys.argv[1])
    print(maxeps)
    mutil_device_run(maxeps)


    #eight_process_run()

    exit()
