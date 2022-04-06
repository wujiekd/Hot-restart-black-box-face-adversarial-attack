import pandas
import numpy as np

def cal_attack_rate():
    f = open('iter_num.txt')
    strs = f.read()
    strs = strs.split('\n')
    #print(strs)

    scores = []
    attack_num = 0
    fail_num = 0
    all_num = 0

    threshold = 0.38
    for i in strs:
        if i.split(';') != ['']:
            all_num +=1
            #print(float(i.split(';')[2]))
            score = float(i.split(';')[2])
            scores.append(score)
            #print(score)
            if score< threshold:
                #print(score)
                attack_num+=1
            elif score>= threshold:
                fail_num+=1

    print("攻击成功:",attack_num/all_num)
    print("攻击失败:",fail_num/all_num)
    print("图像数量:",all_num)
    print("平均余弦相似度:",np.mean(scores))

    return attack_num/all_num,all_num,np.mean(scores)

if __name__ == '__main__':
    cal_attack_rate()