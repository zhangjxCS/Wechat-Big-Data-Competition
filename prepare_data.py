# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
import sys


# 存储数据的根目录
ROOT_PATH = "../data"
# 比赛数据集路径
DATASET_PATH = ROOT_PATH + '/wechat_algo_data1/'
# 训练集
USER_ACTION = DATASET_PATH + "user_action.csv"
FEED_INFO = DATASET_PATH + "feed_info.csv"
FEED_EMBEDDINGS = DATASET_PATH + "feed_embeddings.csv"
# 测试集
TEST_FILE = DATASET_PATH + "test_b.csv"
# 初赛待预测行为列表
ACTION_LIST = ["read_comment", "like", "click_avatar", "forward"]
FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar", "forward", "comment", "follow", "favorite"]
FEA_FEED_LIST = ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id', 'manual_keyword_list', 'machine_keyword_list', 'manual_tag_list', 'machine_tag_list']
# 负样本下采样比例(负样本:正样本)
ACTION_SAMPLE_RATE = {"read_comment": 5, "like": 5, "click_avatar": 5, "forward": 10, "comment": 10, "follow": 10,
                      "favorite": 10}

def process_embed(train):
    feed_embed_array = np.zeros((train.shape[0], 512))
    for i in tqdm(range(train.shape[0])):
        x = train.loc[i, 'feed_embedding']
        if x != np.nan and x != '':
            y = [float(i) for i in str(x).strip().split(" ")]
        else:
            y = np.zeros((512,)).tolist()
        feed_embed_array[i] += y
    temp = pd.DataFrame(columns=[f"embed{i}" for i in range(512)], data=feed_embed_array)
    train = pd.concat((train, temp), axis=1)
    return train

def prepare_data():
    feed_info_df = pd.read_csv(FEED_INFO)
    user_action_df = pd.read_csv(USER_ACTION)[["userid", "date_", "feedid"] + FEA_COLUMN_LIST]
    feed_embed = pd.read_csv(FEED_EMBEDDINGS)
    test = pd.read_csv(TEST_FILE)
    # feed_embed PCA 降维
    feed = np.array(feed_embed.feed_embedding.str.split().tolist())
    print(feed.shape)
    pca = PCA(n_components=int(sys.argv[1]))
    res=pca.fit_transform(feed)
    print(res.shape)
    print(pca.explained_variance_ratio_)
    df = pd.DataFrame(res)
    df['feedid'] = feed_embed['feedid']
    df.columns = [f'embed{b}' for b in range(int(sys.argv[1]))]+['feedid']
    # add feed feature
    train = pd.merge(user_action_df, feed_info_df[FEA_FEED_LIST], on='feedid', how='left')
    test = pd.merge(test, feed_info_df[FEA_FEED_LIST], on='feedid', how='left')
    # add feed_embedding
    test = pd.merge(test, df, on='feedid', how='left')
    train = pd.merge(train, df, on='feedid', how='left')
    test["videoplayseconds"] = np.log(test["videoplayseconds"] + 1.0)
    test.to_csv(ROOT_PATH + f'/test_data.csv', index=False)
    for action in tqdm(ACTION_LIST):
        print(f"prepare data for {action}")
        tmp = train.drop_duplicates(['userid', 'feedid', action], keep='last')
        df_neg = tmp[tmp[action] == 0]
        df_neg = df_neg.sample(frac=1.0 / ACTION_SAMPLE_RATE[action], random_state=42, replace=False)
        df_all = pd.concat([df_neg, tmp[tmp[action] == 1]])
        df_all["videoplayseconds"] = np.log(df_all["videoplayseconds"] + 1.0)
        df_all.to_csv(ROOT_PATH + f'/train_data_for_{action}.csv', index=False)


if __name__ == "__main__":
    prepare_data()