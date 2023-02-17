# Hot-restart-black-box-face-adversarial-attack

我的竞赛代码实现以及我的毕业论文整理~

2021OPPO安全AI挑战赛，比赛链接：
https://security.oppo.com/challenge/rank.html

初赛Rank: 6/3000+ ，复赛Rank: 12/3000+ ，前十保底3w RMB，血亏orz
<!-- 初赛：
![3S}{F4~ DR140)}M%N_P`9S](https://user-images.githubusercontent.com/49955700/161908913-68270e7d-53cf-42df-ab04-4968922579e2.png)

复赛：
![U}PFY 4UX7XU(@NZJ`%1G}U](https://user-images.githubusercontent.com/49955700/161908932-11e32989-8c3e-4fe4-9a99-74fd2b6381b9.png) -->


# 1. 运行方法

两个文件夹，将图片文件夹分成两部分，一部分为亚洲人脸，另一部分为欧洲人脸数据集。

1、攻击亚洲人脸数据集方法如下：

进入Asia文件夹，把原始图片放置./Asia/source_data内，按照以下顺序代码即可：

python process.py

python train.py

python move.py



2、攻击欧美人脸数据集方法如下：

进入Europe文件夹，把原始图片放置./Europe/source_data内，按照以下顺序代码即可：

python process.py

python train.py

python move.py



3、将result_data文件夹内的两个images合并即可





# 2.解决方案

## 2.1数据分析

由于两个数据集分布差异较大，一个为亚洲人脸数据集，另一个为欧美人脸数据集。并且对两类图片进行了一个简单分析，发现亚洲人脸图片的size较小，在150\*150左右，而欧美人脸图片的size较大，在1000\*1000左右。

所以，将这两类数据分别进行不同的攻击（攻击方案差异不大）。

## 2.2攻击方案

### 2.2.1基本攻击框架

考虑到赛题的黑盒场景，以及为了提高对抗样本的迁移性，主要的`基本攻击框架`采用了以下方法(参数设置见具体代码文件）：

1、基于动量的多次迭代FGSM，MI-FGSM[1]

2、输入多样性 input diversity[2]

3、平移不变攻击 Translation-Invariant Attack[3]

4、集成攻击 model ensemble[4]

这里集成的模型均是从Github上开源链接获取，代码里有标注所有模型的Github来源。



### 2.2.2损失函数

针对赛题要求，人脸比对攻击，我们选择的人脸评价指标为`余弦距离`，如下：
![MommyTalk1649227900841](https://user-images.githubusercontent.com/49955700/161914856-a6105535-9b1d-498a-9b6c-538586dc45f3.png)

根据赛题设计，已经设置好了图像配对的规则，并设置了**MS-SSIM** (multi-scale structural similarity)图像质量评价法，因此我们定义的`损失函数`如下：
![MommyTalk1649228513186](https://user-images.githubusercontent.com/49955700/161914955-126088ec-be66-42f2-9805-9f64c28014cb.png)

其中：
![MommyTalk1649228552069](https://user-images.githubusercontent.com/49955700/161915035-154c3bc4-a803-4bf8-9fe5-1627ba9e7c8a.png)
![MommyTalk1649228570546](https://user-images.githubusercontent.com/49955700/161915104-653ca34c-ad34-4f07-8328-b1c34b3aef42.png)
![MommyTalk1649228578355](https://user-images.githubusercontent.com/49955700/161915114-6e7a82a8-2393-49e5-ac0e-a71345a1083b.png)

这个损失函数可以保证在远离原图的基础上，并且远离需要匹配的目标图片，并且使得图像质量评价得分较高。

### 2.2.3改进点

1、针对三通道L无穷评价指标的改进

以前的方案都是设置了一个固定的eps`（扰动阈值）`，比如本次赛题最大eps设置为20，那我们可以设置为15或者10来获得更高的得分，但对于所有图片来说，有的图片比较好攻击，有的图片不好攻击，设置eps过小可能会导致大量图片攻击失败，因此需要的扰动范围是完全不同的。

因此，我们设计了**一种基于分位数的热重启三通道L无穷迭代攻击**，这样eps会因输入而异。

攻击方法如下：

对于每张图片，在迭代攻击时，当攻击余弦相似度小于设置好的阈值`（评价阈值）`时，对于该图片的三通道分别计算98分位数，使用新的eps进行clip，继续攻击，使得满足余弦相似度小于设置好的阈值。

这个攻击方法可以连续使用多次，具体参数看具体代码。



2、针对两类图片size的改进

对于亚洲人脸数据集，size较小，比较符合人脸模型的112\*112输入，因此采用**端到端**的攻击方法，即采用torch的上采样函数，该方法可导，使得梯度攻击可以实现端到端的攻击方案。

对于欧美人脸数据集，size非常大，采用端到端的方案会极度不稳定，因此采用提前resize为112\*112的方案。


3、噪声初始化方法的选择

灵感来源于PGD攻击[5]以及ODI-PGD攻击[6]，初始化往往可以获得一个很好的起点，更有利于攻击成功。

但由于PGD攻击采用的是对eps的均匀分布，初始化噪声较大，因此我们需要更换使用不同的初始方案。

对于亚洲人脸数据集，采用的是**kaiming_uniform初始化**；对于欧美人脸数据集，采用的是**xavier_normal初始化**。



4、Target图像的提取

考虑到赛方在进行线上评测时，可能会使用到检测人脸，再进行对比计算，因此对匹配图片采用mctnn检测得到112\*112的人脸进行计算loss。


# 3.思考

复赛第一阶段排名第6，第二阶段排名第13，发现欧美人脸数据集分布不是很一致，复赛第二阶段的欧美人脸比第一阶段整体size都大了很多，所以第一阶段的算法实现有所下降，希望主办方可以考虑这个问题进行排名评分。


# 4.团队介绍

大四本科生。


# 5.附录

1. Yinpeng Dong, Fangzhou Liao, Tianyu Pang, Hang Su, Jun Zhu, Xiaolin Hu, Jianguo Li; "Boosting Adversarial Attacks With Momentum";The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 9185-9193

2. Cihang Xie, Yuxin Wu, Laurens van der Maaten, Alan L. Yuille, Kaiming He; "Feature Denoising for Improving Adversarial Robustness";The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 501-509 

3. Yinpeng Dong, Tianyu Pang, Hang Su, Jun Zhu; “Evading Defenses to Transferable Adversarial Examples by Translation-Invariant Attacks”.

4. Hansen L K, Salamon P. Neural network ensembles[J]. IEEE transactions on pattern analysis and machine intelligence, 1990, 12(10): 993-1001.

5. Madry A, Makelov A, Schmidt L, et al. Towards deep learning models resistant to adversarial attacks[J]. arXiv preprint arXiv:1706.06083, 2017.

6. Tashiro Y, Song Y, Ermon S. Diversity can be transferred: Output diversification for white-and black-box attacks[J]. Advances in Neural Information Processing Systems, 2020, 33.

   

