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
from getmax import getmax
import shutil
warnings.filterwarnings("ignore")
to_torch_tensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

# 返回人脸识别模型（主干网络）（输出512维的向量）
def get_all_model(model, param, device, proportion):
    m = model([112, 112])
    m.eval()
    m.to(device)

    m.load_state_dict(torch.load(param, map_location=device))
    model_dict = {'model': m, 'proportion': proportion}
    return model_dict


def get_other_model(weight, name,device,value ,proportion):
    m = get_model(name, fp16=value)
    m.eval()
    m.to(device)

    m.load_state_dict(torch.load(weight, map_location=device))
    model_dict = {'model': m, 'proportion': proportion}
    return model_dict


def get_other_model2(name,device ,proportion):
    m = InceptionResnetV1(pretrained='vggface2',device = device).eval()

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
    model_pool.append(get_all_model(IR_50, '../user_data/models/Backbone_IR_50_LFW.pth', device, 1)) #单模 225.52
    model_pool.append(
        get_all_model(IR_101, '../user_data/models/Backbone_IR_101_Batch_108320.pth', device, 1))  # 单模244.621078
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





# path为数据集的单极目录结构
def callike(path):
    device = torch.device('cuda:0')
    model_pool = get_model_pool(device)

    print('----models load over----')
    faces = os.listdir(path)

    vectors_list = []
    for model_dict in model_pool:
        model = model_dict['model']

        vectors = []
        for face_name in faces:
            #print(face)
            origin_img = Image.open(path+'/'+face_name).convert('RGB')
            #origin_img = origin_img.resize((112, 112), Image.NEAREST)
            face = to_torch_tensor(origin_img)

            face = face.unsqueeze_(0).to(device)

            face = F.upsample(face, size=[112, 112], mode='bicubic')
            v1  = l2_norm(model(face)).detach_()  #torch.Size([1, 512])
            vectors.append(v1)
        vectors_list.append(vectors)   #多少个模型就多少个vectors_list
        print(len(vectors_list))

    value = 0

    print('----vectors calculate over----')
    confusion_matrixes = []
    for vectors in vectors_list:
        print("len(vectors)", len(vectors))
        s = torch.FloatTensor(len(vectors), len(vectors))
        for i, vector1 in enumerate(vectors):
            for j, vector2 in enumerate(vectors[i + 1:]):
                #tmp = (vector1 * vector2).sum().item()
                print(torch.square(vector1 - vector2).shape)
                tmp = torch.square(vector1 - vector2).sum().item()
                s[i, j + i + 1] = tmp
                s[j + i + 1, i] = tmp

        print("len(vectors_list)", len(vectors_list))
        for i in range(len(vectors_list)):
            s[i, i] = value
        confusion_matrixes.append(s)

    # In[5]:
    confusion_matrix = confusion_matrixes[0].clone()  #合并多个模型计算的结果
    for tmp in confusion_matrixes[1:]:
        confusion_matrix += tmp

    # In[6]:
    import json  #每个人最多5张图片，获取前6张最相似的肯定有不是同一类的！！！
    value1, index_like1 = torch.max(confusion_matrix, 1)
    for i, j in enumerate(index_like1):
        confusion_matrix[i, j] = value
    value2, index_like2 = torch.max(confusion_matrix, 1)
    for i, j in enumerate(index_like2):
        confusion_matrix[i, j] = value
    value3, index_like3 = torch.max(confusion_matrix, 1)
    for i, j in enumerate(index_like3):
        confusion_matrix[i, j] = value
    value4, index_like4 = torch.max(confusion_matrix, 1)
    for i, j in enumerate(index_like4):
        confusion_matrix[i, j] = value
    value5, index_like5 = torch.max(confusion_matrix, 1)
    for i, j in enumerate(index_like5):
        confusion_matrix[i, j] = value

    value6, index_like6 = torch.max(confusion_matrix, 1)
    a = {}
    for i, face in enumerate(faces):
        #if face.split('_')=='0.jpg':
        print(face)
        a[face] = [faces[index_like1[i]], value1[i].item() / len(model_pool), faces[index_like2[i]], value2[i].item() / len(model_pool),
                   faces[index_like3[i]], value3[i].item() / len(model_pool),faces[index_like4[i]], value4[i].item() / len(model_pool),
                   faces[index_like5[i]], value5[i].item() / len(model_pool),faces[index_like6[i]], value6[i].item() / len(model_pool)]
    f = open("likelihood_images.json", "w")
    f.write(json.dumps(a))
    f.close()

# path为数据集的单极目录结构
def compute(path):
    device = torch.device('cuda:0')
    model_pool = get_model_pool(device)

    print('----models load over----')
    faces = os.listdir(path)

    vectors_list = []
    for model_dict in model_pool:
        model = model_dict['model']

        vectors = []
        for face_name in faces:
            # print(face)
            origin_img = Image.open(path + '/' + face_name).convert('RGB')
            # origin_img = origin_img.resize((112, 112), Image.NEAREST)
            face = to_torch_tensor(origin_img)

            face = face.unsqueeze_(0).to(device)

            face = F.upsample(face, size=[112, 112], mode='bicubic')
            v1 = l2_norm(model(face)).detach_()  # torch.Size([1, 512])

            #v1 = model(face).detach_()
            vectors.append(v1)
        vectors_list.append(vectors)  # 多少个模型就多少个vectors_list
        print(len(vectors_list))

    value = 0

    print('----vectors calculate over----')
    confusion_matrixes = []
    for vectors in vectors_list:
        print("len(vectors)", len(vectors))
        s = torch.FloatTensor(len(vectors), len(vectors))
        for i, vector1 in enumerate(vectors):
            for j, vector2 in enumerate(vectors[i + 1:]):
                tmp = (vector1 * vector2).sum().item()
                #print(torch.square(vector1 - vector2).shape)
                #tmp = torch.square(vector1 - vector2).sum().item()
                s[i, j + i + 1] = tmp
                s[j + i + 1, i] = tmp

        print("len(vectors_list)", len(vectors_list))
        for i in range(len(vectors_list)):
            s[i, i] = value
        confusion_matrixes.append(s)


    # In[5]:
    confusion_matrix = confusion_matrixes[0].clone()  # 合并多个模型计算的结果
    for tmp in confusion_matrixes[1:]:
        confusion_matrix += tmp

    ans = []
    save = {}
    for i in range(1000):

        have =[]
        if faces[i].split('_')[1] == '0.jpg' :
            print(i)
            print(faces[i])
            for j in range(1000):
                if faces[i].split('_')[0] == faces[j].split('_')[0]:
                    if i==j:
                        have.append(faces[j])
                        have.append(0)
                    else:
                        print(j)
                        print(confusion_matrix[i][j]/len(model_pool))
                        have.append(faces[j])
                        num = (confusion_matrix[i][j]/len(model_pool)).item()
                        sco = num #1 / (1 + num)
                        have.append(sco)
                        ans.append(sco)

            print(have)
            save[faces[i].split('.')[0]+'.jpg'] = have
        else:
            continue
    f = open("余弦距离.json", "w")
    f.write(json.dumps(save))
    f.close()
    print(ans)
    print(np.max(np.array(ans)))
    print(np.min(np.array(ans)))

def get():
    edu = json.load(open("余弦距离.json"))
    print(edu)

    all1 = []
    all2 = []
    all = []

    set_value = 0.8
    for key in edu.keys():
        if key.split('_')[1] == '0.jpg':
            pass
        else:
            continue

        ll = edu[key]
        for j in range(int(len(ll)/2)):
            if ll[2*j].split('_')[1] == '1.jpg':  #和1比较，不满足把第0张存储下来
                all.append(ll[2*j+1])
                if ll[2 * j + 1] < set_value:#0.8
                    all1.append(ll[2 * j + 1])
                else:
                    all2.append(ll[2 * j + 1])


                    source = '../result_data/images/' + key.split('_')[0] +'/' +key
                    deter2 = '../result_data/new/' + key
                    shutil.copyfile(source, deter2)


            if ll[2 * j].split('_')[1] == '0.jpg': #0直接跳过
                pass


            else:  #其他四张跟0比较，把四张中的有问题的拿下来
                all.append(ll[2*j+1])

                if ll[2*j+1] < set_value:
                    all1.append(ll[2*j+1])
                else:
                    all2.append(ll[2*j+1])

                    source = '../result_data/images/' + str(ll[2*j]).split('_')[0] +'/' + str(ll[2*j])
                    deter2 = '../result_data/new/' + str(ll[2*j])
                    shutil.copyfile(source, deter2)


    print(np.max(np.array(all)))
    print(np.min(np.array(all)))
    print(len(all))
    print(np.quantile(np.array(all),0.8))
    print(len(all1))
    print(len(all2))

    #print(all1)
#测试小于0。6的攻击到-0.35效果如何

# path为数据集的单极目录结构
def getvector(path):
    device = torch.device('cuda:0')
    model_pool = get_model_pool(device)

    print('----models load over----')
    faces = os.listdir(path)



    vectors = np.zeros([200,512])

    for face_name in faces:
        v1 = np.zeros([1, 512])
        # print(face)
        origin_img = Image.open(path + '/' + face_name).convert('RGB')
        # origin_img = origin_img.resize((112, 112), Image.NEAREST)
        face = to_torch_tensor(origin_img)

        face = face.unsqueeze_(0).to(device)

        face = F.upsample(face, size=[112, 112], mode='bicubic')

        for model_dict in model_pool:
            model = model_dict['model']
            vec =  l2_norm(model(face)).detach()  #torch.Size([1, 512])
            #print(vec.shape)

            v1 += vec.data.cpu().detach().numpy()

        v1 = v1/len(model_pool)

        print(int(face_name.split('_')[0]))
        vectors[int(face_name.split('_')[0]),:] = v1

    print(vectors)
    np.save("./vectors.npy", vectors)


if __name__ == '__main__':

    #callike('../source_data/adv_data112')
    #getmax()

    #compute("../source_data/adv_data112")

    get()


