# 1.背景

清华大学和腾讯联合举办的2021中国高校计算机大赛——微信大数据挑战赛（[https://algo.weixin.qq.com](https://algo.weixin.qq.com)），提供了微信视频号的用户行为、视频基本信息等数据，要求参赛者运用算法预测测试集上用户对视频的行为（包括查看评论、点赞、点击头像、转发等）的发生概率。

# 2.模型

大赛官方为参赛者提供了多个模型的baseline，我主要参考了wide & deep（[https://github.com/WeChat-Big-Data-Challenge-2021/WeChat_Big_Data_Challenge.git](https://github.com/WeChat-Big-Data-Challenge-2021/WeChat_Big_Data_Challenge.git)）和deepfm（[https://github.com/dpoqb/wechat_big_data_baseline_pytorch](https://github.com/dpoqb/wechat_big_data_baseline_pytorch)）两个模型的baseline，并在deepfm模型的baseline基础上加以改进。

改进的思路主要有两点：一是从数据特征入手，在baseline的6项特征（用户id，视频id，视频作者id，视频时长，视频bgm的id，视频bgm歌手的id）基础上，增加了多模态内容理解特征、关键词和标签作为模型的输入特征，其中多模态内容理解特征采用PCA从512维降至64维，人工关键词、机器关键词和人工标签均取第一个，机器标签取概率最大的；二是调节超参数，如线性部分的l2正则化、dnn部分的l2正则化、embedding向量的l2正则化，以及结合10折交叉验证，选择最优的迭代次数。

代码见prepare_data.py和deepfm.py两个文件，其中prepare_data.py文件用于数据预处理，生成训练集和测试集，deepfm.py文件用于模型训练与预测。

运行示例如下：

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

# 3.结果

上述deepfm模型在a榜的得分为0.659.

| 得分     | 查看评论 | 点赞     | 点击头像 | 转发     |
| -------- | -------- | -------- | -------- | -------- |
| 0.659452 | 0.640185 | 0.626303 | 0.728065 | 0.698743 |

将该模型与队友的lgb、deepfm模型（有所不同）融合，a榜得分有了进一步提升。

| 得分     | 查看评论 | 点赞     | 点击头像 | 转发    |
| -------- | -------- | -------- | -------- | ------- |
| 0.668285 | 0.649172 | 0.638987 | 0.733406 | 0.70239 |

最终b榜得分为0.663，排名175.

| 得分     | 查看评论 | 点赞     | 点击头像 | 转发     |
| -------- | -------- | -------- | -------- | -------- |
| 0.663659 | 0.63827  | 0.635107 | 0.739283 | 0.699627 |

