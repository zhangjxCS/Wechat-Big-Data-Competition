# 微信大数据竞赛

## 1.背景

清华大学和腾讯联合举办的2021中国高校计算机大赛——[微信大数据挑战赛](https://algo.weixin.qq.com)。本次比赛基于脱敏和采样后的数据信息，对于给定的一定数量到访过微信视频号“热门推荐”的用户，根据这些用户在视频号内的历史n天的行为数据，通过算法在测试集上预测出这些用户对于不同视频内容的互动行为（包括点赞、点击头像、收藏、转发等）的发生概率。本次比赛以多个行为预测结果的加权uAUC值进行评分。

## 2.数据

所有数据的下载可见[官方网站](https://algo.weixin.qq.com/problem-description)，主要数据包括以下：

• feed_info.csv: 视频（简称为feed）的基本信息和文本、音频、视频等多模态特征

• user_action.csv: 用户在视频号内一段时间内的历史行为数据（包括停留时长、播放时长和各项互动数据）

• feed_embeddings.csv: 基于视频信息、视频文字信息、语音信息构建的embedding矩阵

• test_a.csv test_b.csv: A榜与B榜测试集

## 3.特征工程

推荐系统中常用的特征主要有以下三个部分

• 用户信息：包括用户手机号、设备、年龄、性别等

• 视频信息：包括视频描述、字母、关键词、标签、背景音乐等信息

• 用户-视频交互信息：包括用户对视频的点赞、关注、停留时长等信息

常用的构造特征方法：

• 行为统计特征：对于时间序列特征，通常使用滑动窗口方法构造统计特征；通常的模式是时间\*行为*统计量，构造近1周、近5天的用户行为（点赞、关注、评论等）的统计量（计数、求和、最小值、最大值、均值等）；在计算统计量之后，可以对统计量进行二次衍生，计算比例，例如用户点赞次数占比等

• 交互统计特征：通过滑动窗口方法构造用户对视频作者、视频bgm、视频标签等的交互特征次数和比例，挖掘用户对视频作者、视频bgm等信息的偏好程度

• 文本特征：对视频的标签、关键词等信息当作离散变量进行处理；处理方法包括Label Encoding, Onehot Encoding, Target Encoding, TF-IDF, Embedding等

## 4.模型

在广告推荐领域常用的模型主要包括树模型、deepFM模型和Wide & Deep模型，官方的baseline提供了Wide & Deep模型的[代码](https://github.com/WeChat-Big-Data-Challenge-2021/WeChat_Big_Data_Challenge.git)，后续我们构建了基于决策树模型和GBDT思路的LightGBM模型，以及deepfm两个模型。

### LightGBM模型

LightGBM是基于决策树的集成模型，采用GBDT在前树拟合的残差上继续建树的方法进行。使用构造的特征分别对四个互动行为特征建树模型，在交叉验证集上确定最优迭代次数，在测试集上给出预测值

### Wide & Deep模型

下面是wide&deep模型的结构图，由左边的wide部分(一个简单的线性模型)，右边的deep部分(一个典型的DNN模型)。在构建模型时，根据使用场景选择部分特征放在Wide部分，部分特征放在Deep部分。Wide & Deep模型可以平衡模型的记忆能力与泛化能力，推荐结果更精确

![image-20200910214310877](http://datawhale.club/uploads/default/optimized/1X/17d8be55548582135c76bc5e6a6c50c896a9fb14_2_690x170.png)

### DeepFM模型

和Wide & Deep的模型类似，DeepFM模型同样由浅层模型和深层模型联合训练得到。不同点主要有以下两点：

• wide模型部分由LR替换为FM。FM模型具有自动学习交叉特征的能力，避免了原始Wide & Deep模型中浅层部分人工特征工程的工作

• 共享原始输入特征。DeepFM模型的原始特征将作为FM和Deep模型部分的共同输入，保证模型特征的准确与一致

![img](https://pic2.zhimg.com/80/v2-a893a331c3556046be1be7771b2cb1a9_720w.jpg)



## 5.代码

### LightGBM

```python
# 读入数据，使用LightGBM模型预测结果
!python lightgbm.py
```

### DeepFM

```python
# 数据预处理，传递的参数表示PCA降维的维度，如64维
!python prepare_data.py 64
# 模型训练与预测，传递的参数分别表示PCA降维的维度、线性部分的l2正则化、embedding向量的l2正则化、dnn部分的l2正则化、四类行为的迭代次数、第k折
!python deepfm.py 64 0.2 0.2 0.2 7 7 19 8 1
!python deepfm.py 64 0.2 0.2 0.2 7 6 19 8 2
!python deepfm.py 64 0.2 0.2 0.2 7 6 18 8 3
!python deepfm.py 64 0.2 0.2 0.2 7 6 19 9 4
!python deepfm.py 64 0.2 0.2 0.2 8 8 17 9 5
!python deepfm.py 64 0.2 0.2 0.2 7 6 17 10 6
!python deepfm.py 64 0.2 0.2 0.2 7 6 17 10 7
!python deepfm.py 64 0.2 0.2 0.2 8 6 17 9 8
!python deepfm.py 64 0.2 0.2 0.2 8 8 15 9 9
!python deepfm.py 64 0.2 0.2 0.2 6 6 20 8 10
# 合并预测结果
import pandas as pd
df = pd.read_csv('submit_base_deepfm_1valid_b榜.csv')
for i in range(2,11):
  print(i)
  df = df + pd.read_csv(f'submit_base_deepfm_{i}valid_b榜.csv')
df = df / 10
df.to_csv('submit_base_deepfm_10fold_b榜.csv', index=False)
```

## 6.结果

lightgbm模型在a榜得分0.655

deepfm模型在a榜的得分为0.659将deepFM模型与lgb融合，a榜得分有了进一步提升。

| 得分     | 查看评论 | 点赞     | 点击头像 | 转发    |
| -------- | -------- | -------- | -------- | ------- |
| 0.668285 | 0.649172 | 0.638987 | 0.733406 | 0.70239 |

最终b榜得分为0.663，排名175.

| 得分     | 查看评论 | 点赞     | 点击头像 | 转发     |
| -------- | -------- | -------- | -------- | -------- |
| 0.663659 | 0.63827  | 0.635107 | 0.739283 | 0.699627 |

