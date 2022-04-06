'''
复赛baselie 要修改loss
！！！！！！！！！！！！times = 3 break_threshold = -0.15 直接3代不就完事咯
初赛：436.75 819.8910111188887 1173.033   #the best 586左右

复赛：468.73333333333335 835.5000481009482  1510.529175
'''


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
warnings.filterwarnings("ignore")
to_torch_tensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

maxeps = 20.0
times = 3

def input_diversity(image, prob=0.5, low=112, high=128):
    if random.random() < prob:
        #image = F.upsample(image, size=[low, low], mode='nearest')  # bilinear
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
    noise = torch.nn.init.xavier_normal(noise, gain=1)
    #noise = torch.nn.init.kaiming_uniform(noise)
    #noise = torch.nn.init.uniform_(noise, a=-maxeps / 255 * 2, b=maxeps / 255 * 2)
    # noise = torch.zeros_like(input) #不初始化噪声
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
    print('cao')
    m = MobileFaceNet(512)
    m.eval()
    m.to(device)

    m.load_state_dict(torch.load(param, map_location=device))
    model_dict = {'model': m, 'proportion': proportion}
    return model_dict


# 返回模型池
def get_model_pool(device):
    model_pool = []
    # 以下为同一个文件
    # model_pool.append(get_all_model(IR_50, '../user_data/models/backbone_ir50_ms1m_epoch120.pth', device, 1))  #单模 155.708878，https://github.com/ZhaoJ9014/face.evoLVe.PyTorch?spm=5176.21852664.0.0.7a6570e7wfEMmH&file=face.evoLVe.PyTorch#Model-Zoo
    # model_pool.append(get_all_model(IR_50, '../user_data/models/Backbone_IR_50_LFW.pth', device, 1)) #单模 225.52
    #model_pool.append(get_all_model(IR_101, '../user_data/models/Backbone_IR_101_Batch_108320.pth', device, 1)) #单模244.621078
    # model_pool.append(get_all_model(IR_152, '../user_data/models/Backbone_IR_152_MS1M_Epoch_112.pth', device, 1)) #单模 159.544693,https://github.com/ZhaoJ9014/face.evoLVe.PyTorch?spm=5176.21852664.0.0.7a6570e7wfEMmH&file=face.evoLVe.PyTorch#Model-Zoo
    model_pool.append(get_all_model(IR_50, '../user_data/models/Backbone_IR_50_LFW_ADV_TRAIN.pth', device,
                                    1))  # 居然有对抗训练的模型，得试试#单模227.112106，对抗效果一般般
    # model_pool.append(get_all_model(IR_50, '../user_data/models/backbone_ir50_asia.pth', device,1))  # 131.105804 Aisa数据集，https://github.com/ZhaoJ9014/face.evoLVe.PyTorch?spm=5176.21852664.0.0.7a6570e7wfEMmH&file=face.evoLVe.PyTorch#Model-Zoo
    model_pool.append(get_all_model(IR_SE_50, '../user_data/models/model_ir_se50.pth', device,
                                    1))  # 250.737747  https://github.com/TreB1eN/InsightFace_Pytorch

    # model_pool.append(get_other_model('../user_data/models/ms1mv3_arcface_r100_fp16/backbone.pth', 'r100',device,True,1)) #94.943787
    # model_pool.append(get_other_model('../user_data/models/ms1mv3_arcface_r2060/backbone.pth', 'r2060', device, False, 1))#88.056427
    # model_pool.append(get_other_model('../user_data/models/glint360k_cosface_r100_fp16_0.1/backbone.pth', 'r100', device, True,1))  #94.654289

    model_pool.append(get_other_model2('vggface2', device, 1))  # 287.170624 https://github.com/timesler/facenet-pytorch
    model_pool.append(
        get_other_model2('casia-webface', device, 1))  # 290.522003 https://github.com/timesler/facenet-pytorch

    # 以下为另一个文件
    model_pool.append(get_other_model3('../user_data/models/model_mobilefacenet.pth', device,
                                       1))  # 单模192.300568，https://github.com/ZhaoJ9014/face.evoLVe.PyTorch?spm=5176.21852664.0.0.7a6570e7wfEMmH&file=face.evoLVe.PyTorch#Model-Zoo

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
    noise = tmp_noise #*   # noise = gaussian_blur(tmp_noise)   #高斯平滑
    loss1 = 0
    score1 = 0
    score2 = 0

    adv_image = origin_img + noise
    adv_image_pool = input_diversity(adv_image)
    for model_dict in model_pool:
        model = model_dict['model']
        proportion = model_dict['proportion']
        v1 = l2_norm(model(adv_image_pool))  # 对抗样本向量
        v2_1 = l2_norm(model(origin_img)).detach_()  # 原图向量
        v2_2 = l2_norm(model(target_img)).detach_()  # 目标图片向量

        # 用的余弦相似度，越大越相似
        tmp1 = (v1 * v2_1).sum()  # 对抗样本 与 原图 的向量的内积
        tmp2 = (v1 * v2_2).sum()  # 对抗样本 与 目标图片 的向量的内积

        loss1 += (tmp1 + tmp2)* proportion #(tmp1 - tm2) * proportion  # 只采用非定向，不使用定向，集成模型
        score1 += (tmp1.item() + tmp2.item()) * proportion / 2
        #score2 += tmp2.item() * proportion

    m = pytorch_msssim.MSSSIM()

    msssim = m(origin_img, adv_image).item()  # 正则化项,越大越好，可以优化
    loss3 = (msssim - 0.8) * 5

    # print(loss1)
    # print(loss3)
    all_loss = loss1 - loss3

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

    loss1_v_gauss = F.conv2d(now_loss1_v, gaussian_blur, padding=(3 - 1) // 2, groups=3)  # 动量可以试着加在平滑后面

    L1_grad_norms = loss1_v_gauss.view(1, -1).norm(p=1, dim=1)  # 使用L1进行正则化
    L1_norm_loss1_v = loss1_v_gauss.div(L1_grad_norms.view(-1, 1, 1, 1))



    loss1_v = L1_norm_loss1_v + loss1_v * momentum  # 使用动量项，利用loss1更新扰动
    # print(torch.max(tmp_noise.grad))


    grad_norms = loss1_v.view(1, -1).norm(p=float('inf'), dim=1)  # 使用L无穷进行正则化
    norm_loss1_v = loss1_v.div(grad_norms.view(-1, 1, 1, 1))

    tmp_noise = tmp_noise.detach() - lr * norm_loss1_v  # 不用正则化

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
    return tmp_noise , score1, score2, loss1_v


# 多步迭代, 调用单步迭代函数
def noise_iter(model_pool, origin_img, target_img, gaussian_blur, device,origin_name,break_threshold):
    learning_rate = maxeps / 255 * 2 / 20  # 可以调整,步长为4，因为图片放缩了，步长也要放缩

    momentum = 0.9  # 可以调整
    tmp_noise = get_init_noise(device, origin_img)  # 初始化噪声
    tmp_noise_save = tmp_noise.clone()  # 存储上一次进入判断的噪声
    index = 0
    loss1_v = 0
    eps1 = maxeps / 255 * 2  # 起始最大eps
    eps2 = maxeps / 255 * 2
    eps3 = maxeps / 255 * 2
    flag = 0
    while True:

        index += 1


        tmp_noise, score1, score2, loss1_v = iter_step(tmp_noise, origin_img, target_img, gaussian_blur, model_pool,
                                                       index, loss1_v, momentum, learning_rate, eps1, eps2, eps3)

        if score1 < break_threshold and flag != times:
            tmp_noise_save  = tmp_noise.clone() # 存储上一次噪声
            noise = tmp_noise[0].cpu().numpy()
            noise = noise.swapaxes(0, 1).swapaxes(1, 2)


            value = 1 - 0.02 * (flag+1) #0.98
            eps1 = np.quantile(np.abs(noise[:, :, 0]), value)
            eps2 = np.quantile(np.abs(noise[:, :, 1]), value)
            eps3 = np.quantile(np.abs(noise[:, :, 2]), value)

            tmp_noise[0][0] = tmp_noise[0][0].clamp_(-eps1, eps1)  # clip
            tmp_noise[0][1] = tmp_noise[0][1].clamp_(-eps2, eps2)  # clip
            tmp_noise[0][2] = tmp_noise[0][2].clamp_(-eps3, eps3)  # clip

            tmp_noise = (tmp_noise + origin_img).clamp_(-1, 1) - origin_img

            print(origin_name ,score1)
            tmp_noise, score1, score2, loss1_v = iter_step(tmp_noise, origin_img, target_img, gaussian_blur, model_pool,
                                                           index, loss1_v, momentum, learning_rate, eps1, eps2, eps3)

            index = 1
            flag += 1
        yield tmp_noise , score1, score2, flag,tmp_noise_save


# 计算单个对抗样本
def cal_adv(origin_name, target_name, model_pool, gaussian_blur, device,break_threshold):
    origin_img = Image.open(origin_name).convert('RGB')
    origin_img = origin_img.resize((112, 112), Image.BICUBIC)
    origin_img = to_torch_tensor(origin_img)
    origin_img = origin_img.unsqueeze_(0).to(device)

    target_img = Image.open(target_name).convert('RGB')
    target_img = target_img.resize((112, 112), Image.BICUBIC)
    target_img = to_torch_tensor(target_img)
    target_img = target_img.unsqueeze_(0).to(device)




    max_iterations = 300
    generator = noise_iter(model_pool, origin_img, target_img, gaussian_blur, device,origin_name,break_threshold)
    scores = 0
    i = 0

    while True:
        tmp_noise, score1, score2, flag,tmp_noise_save = next(generator)
        scores = score1
        i += 1

        if i >= max_iterations:
            if flag == 0:
                f = open('hard2.txt', 'a')
                f.write(origin_name + ';' + target_name + ';' + str(scores) + '\n')
                f.close()
                print('困难样本：origin img is %s, target img is %s, iter %d, socre is %0.3f'
                      % (origin_name.split('/')[-1], target_name.split('/')[-1], i, scores))
            else:
                tmp_noise = tmp_noise_save
                print('origin img is %s, 保存上一次攻击，第 %d 次攻击'
                      % (origin_name.split('/')[-1], flag))


            break


        if score1 <  break_threshold  and flag == times:

            print('origin img is %s, target img is %s, iter %d, socre is %0.3f'
                  % (origin_name.split('/')[-1], target_name.split('/')[-1], i, scores))

            break

    return tmp_noise, i



def attack(face_path, origin_name, tar_path, model_pool, gaussian_blur, device, break_threshold):
    # 生成当前图像的对抗样本
    noise, iter_num = cal_adv(os.path.join(face_path, origin_name),
                              tar_path,
                              model_pool,
                              gaussian_blur,
                              device,
                              break_threshold)
    # print(noise.shape)
    noise = torch.round(noise * 127.5)[0].cpu().numpy()
    # print(noise.shape)
    noise = noise.swapaxes(0, 1).swapaxes(1, 2)
    noise = noise.clip(-maxeps, maxeps)
    """
    value = 0.98
    #加入定向，看是否可以提升成功率
    #value=1.0 141.06666666666666 443.21472555398935 474.082855  81%
    max_0 = np.quantile(np.abs(noise[:, :, 0]), value)#value=0.95 273.15 453.3877000212669 544.32 75%
    max_1 = np.quantile(np.abs(noise[:, :, 1]), value)
    max_2 = np.quantile(np.abs(noise[:, :, 2]), value)

    """
    max_0 = np.abs(noise[:, :, 0]).max()
    max_1 = np.abs(noise[:, :, 1]).max()
    max_2 = np.abs(noise[:, :, 2]).max()  # size : (113, 113, 3) ，查看噪声范围

    origin_img = Image.open(os.path.join(face_path, origin_name)).convert('RGB')
    raw_img = np.array(origin_img, dtype=float)
    h = np.array(origin_img).shape[0]
    w = np.array(origin_img).shape[1]

    # 先resize噪声，再clip，最后add,这种方法非常不可行，才8分，服了！！
    """
    noise = Image.fromarray(np.uint8(noise))  #转换为PIL格式
    noise = noise.resize((h, w), Image.BICUBIC)  #resize噪声
    noise = np.array(noise, dtype=float) #再转回去
    noise[:, :, 0] = np.clip(noise[:, :, 0], -max_0, max_0)
    noise[:, :, 1] = np.clip(noise[:, :, 1], -max_1, max_1)
    noise[:, :, 2] = np.clip(noise[:, :, 2], -max_2, max_2)


    origin_img = np.array(origin_img, dtype=float)
    numpy_adv_sample = (origin_img + noise).clip(0, 255)
    adv_sample = Image.fromarray(np.uint8(numpy_adv_sample))  # 转换为PIL格式

    #先添加噪声，再resize，最后clip
    """
    origin_img = origin_img.resize((112, 112), Image.BICUBIC)  # 添加噪声
    origin_img = np.array(origin_img, dtype=float)
    numpy_adv_sample = (origin_img + noise).clip(0, 255)
    adv_sample = Image.fromarray(np.uint8(numpy_adv_sample))

    adv_sample = adv_sample.resize((h, w), Image.BICUBIC)  # 其实这里还要加一步，控制resize后的像素值是否溢出

    adv_sample = np.array(adv_sample, dtype=float)  # 再转回去
    adv_sample[:, :, 0] = np.clip(adv_sample[:, :, 0], raw_img[:, :, 0] - max_0, raw_img[:, :, 0] + max_0)
    adv_sample[:, :, 1] = np.clip(adv_sample[:, :, 1], raw_img[:, :, 1] - max_1, raw_img[:, :, 1] + max_1)
    adv_sample[:, :, 2] = np.clip(adv_sample[:, :, 2], raw_img[:, :, 2] - max_2, raw_img[:, :, 2] + max_2)
    adv_sample.clip(0, 255)
    adv_sample = Image.fromarray(np.uint8(adv_sample))

    # 保存图像
    if os.path.exists('../result_data/adv_images/') is False:
        os.mkdir('../result_data/adv_images/')
    jpg_img = '../result_data/adv_images/' + origin_name
    png_img = jpg_img.replace('jpg', 'png')
    adv_sample.save(png_img)
    os.rename(png_img, jpg_img)  # 这样可以达到无损～

    # 读取图片进行评分
    # score评价
    ori_img = Image.open(os.path.join(face_path, origin_name)).convert('RGB')
    adv_img = Image.open(jpg_img).convert('RGB')

    score1 = getscore1(ori_img, adv_img)
    score2 = getscore2(ori_img, adv_img)
    print('%s noise score1 is %.4f' % (origin_name, score1))
    print('%s noise score2 is %.4f' % (origin_name, score2))

    ori_img = ori_img.resize((112, 112), Image.BICUBIC)  # 添加噪声
    adv_img = adv_img.resize((112, 112), Image.BICUBIC)  # 添加噪声

    # 读取图片进行评分
    origin_img = to_torch_tensor(ori_img)
    origin_img = origin_img.unsqueeze_(0).to(device)
    adv_img = to_torch_tensor(adv_img)
    adv_img = adv_img.unsqueeze_(0).to(device)

    target_img = Image.open(tar_path).convert('RGB')
    target_img = target_img.resize((112, 112), Image.BICUBIC)
    target_img = to_torch_tensor(target_img)
    target_img = target_img.unsqueeze_(0).to(device)

    score3_1 = 0
    score3_2 = 0
    for model_dict in model_pool:
        model = model_dict['model']
        proportion = model_dict['proportion']
        v1 = l2_norm(model(adv_img))  # 对抗样本向量
        v2_1 = l2_norm(model(origin_img)).detach_()  # 原图向量
        v2_2 = l2_norm(model(target_img)).detach_()  # 原图向量

        # 用的余弦相似度，越大越相似
        tmp1 = (v1 * v2_1).sum()  # 对抗样本 与 原图 的向量的内积
        tmp2 = (v1* v2_2).sum()
        score3_1 += tmp1.item() * proportion
        score3_2 += tmp2.item() * proportion
    print('%s noise score3_1 is %.4f' % (origin_name, score3_1))
    print('%s noise score3_2 is %.4f' % (origin_name, score3_2))

    f = open('iter_num.txt', 'a')
    f.write(jpg_img + ';' + str(score1) + ';' + str(score2) + ';' + str(score3_1) + ';' + str(score3_2) + '\n')
    f.close()
    return score1,score2,score3_1,score3_2

# 单进程运行
def one_process_run(people_list, model_pool, device, face_path):
    if os.path.exists('hard2.txt'):
        os.remove('hard2.txt')
    if os.path.exists('iter_num.txt'):
        os.remove('iter_num.txt')

    #likelihood = json.load(open("maxlikelihood_images.json"))
    kernel = gkern(3, 3).astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 1)
    gaussian_blur = torch.Tensor(stack_kernel).to(device)

    print(len(people_list))

    for origin_name in people_list:

        if origin_name.split('_')[1] == "0.jpg":
            name = origin_name.split('_')[0] + '_1.jpg'
        else:
            name = origin_name.split('_')[0] + '_0.jpg'

        #name = likelihood[origin_name]  # 目标图片
        tar_path = os.path.join("../source_data/adv_data", name)  # 自己类的第一张图片

        # 生成当前图像的对抗样本
        break_threshold = -0.15
        score1,score2,score3_1,score3_2 = attack(face_path, origin_name, tar_path, model_pool, gaussian_blur, device,
                        break_threshold)





# Utilize one GPU to generate an adversarial sample with four processing
def one_device_run(p_pool, people_list, device, face_path):
    four_process = True
    model_pool = get_model_pool(device)
    model_pool = normal_model_proportion(model_pool)
    print('----model load over----')
    if four_process:
        for model_dict in model_pool:
            model_dict['model'].share_memory()
        """
        p_pool.apply_async(one_process_run, args=(people_list[:len(people_list) // 3], model_pool, device, face_path))
        p_pool.apply_async(one_process_run, args=(
            people_list[len(people_list) // 3:len(people_list) // 3 * 2], model_pool, device, face_path))

        p_pool.apply_async(one_process_run, args=(
            people_list[len(people_list) // 3 *2:], model_pool, device, face_path))
        """
        p_pool.apply_async(one_process_run, args=(people_list[:len(people_list) // 8], model_pool, device, face_path))
        p_pool.apply_async(one_process_run, args=(
        people_list[len(people_list) // 8:len(people_list) // 8 * 2], model_pool, device, face_path))
        p_pool.apply_async(one_process_run, args=(
        people_list[len(people_list) // 8 * 2: len(people_list) // 8 * 3], model_pool, device, face_path))
        p_pool.apply_async(one_process_run, args=(
        people_list[len(people_list) // 8 * 3:len(people_list) // 8 * 4], model_pool, device, face_path))
        p_pool.apply_async(one_process_run, args=(
        people_list[len(people_list) // 8 * 4:len(people_list) // 8 * 5], model_pool, device, face_path))
        p_pool.apply_async(one_process_run, args=(
            people_list[len(people_list) // 8 * 5:len(people_list) // 8 * 6], model_pool, device, face_path))
        p_pool.apply_async(one_process_run, args=(
            people_list[len(people_list) // 8 * 6: len(people_list) // 8 * 7], model_pool, device, face_path))
        p_pool.apply_async(one_process_run,
                           args=(people_list[len(people_list) // 8 * 7:], model_pool, device, face_path))


    else:
        p_pool.apply_async(one_process_run, args=(people_list, model_pool, device))


def mutil_device_run():
    face_path = "../source_data/adv_data"

    # device_list 是显卡列表， task_num_list 是每个设备生成对抗样本数量
    # device_list = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']
    # task_num_list = [125, 125, 125, 125]

    device_list = ['cuda:2', 'cuda:3']
    task_num_list = [250, 250]
    if len(task_num_list) != len(device_list):
        raise 'task_num_list is not same as device_list!'
    if np.array(task_num_list).sum() != 500:
        raise 'imgs num is not 500!'

    faces = os.listdir(face_path)

    # 使用多进程实现多显卡同时运行， 也可以实现单显卡的加速
    start_index = 0
    p_pool = torch.multiprocessing.Pool()
    for i, device in enumerate(device_list):
        one_device_run(p_pool, faces[start_index:start_index + task_num_list[i]], device, face_path)
        start_index += task_num_list[i]
    p_pool.close()
    p_pool.join()


# 单GPU八进程
def eight_process_run():
    face_path = "../source_data/adv_data"

    # device_list 是显卡列表， task_num_list 是每个设备生成对抗样本数量
    # device_list = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']
    # task_num_list = [125, 125, 125, 125]

    device_list = ['cuda:0']
    task_num_list = [500]
    if len(task_num_list) != len(device_list):
        raise 'task_num_list is not same as device_list!'
    if np.array(task_num_list).sum() != 500:
        raise 'imgs num is not 500!'

    faces = os.listdir(face_path)

    # 使用多进程实现多显卡同时运行， 也可以实现单显卡的加速
    start_index = 0
    p_pool = torch.multiprocessing.Pool()
    for i, device in enumerate(device_list):
        one_device_run(p_pool, faces[start_index:start_index + task_num_list[i]], device, face_path)
        start_index += task_num_list[i]
    p_pool.close()
    p_pool.join()





# 单进程运行
def one_process_run_solo():
    face_path = "../source_data/adv_data"  # 攻击图像的地址
    if os.path.exists('hard2.txt'):
        os.remove('hard2.txt')
    if os.path.exists('iter_num.txt'):
        os.remove('iter_num.txt')

    device = torch.device('cuda:0')
    model_pool = normal_model_proportion(get_model_pool(device))
    people_list = os.listdir(face_path)

    one_process_run(people_list, model_pool, device, face_path)


if __name__ == '__main__':
    e1 = time.time()
    #one_process_run_solo
    #mutil_device_run()

    eight_process_run()
    e2 = time.time()
    print ("并行执行时间：", int(e2 - e1))
    exit()
