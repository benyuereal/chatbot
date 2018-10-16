from time import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.datasets import fetch_20newsgroups

import numpy as np
import os

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(", ".join([feature_names[i]
                         for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()
# 打印最常见的几个词
def main():
    """打印每个话题里面"n_top_words"个最常见的词语
    """


    # 获取 20newsgroups 数据集
    print("读取数据 ...")
    t0 = time()
    dataset = fetch_20newsgroups(shuffle=True, random_state=1,
                                 remove=('headers', 'footers', 'quotes'))
    print("完成，共耗时 %0.3f 秒." % (time() - t0))

    n_samples = 2000
    # 共有11314篇文本， 我们使用其中2000做实验演示
    print("共有%d篇文本， 我们使用其中%d做实验演示" % (len(dataset['data']), n_samples))
    # 选取前面2000条
    data_samples = dataset.data[:n_samples]
    # 打印几篇样本
    for i in range(3):
        print("%s\n" % dataset['data'][i].replace('\n', ''))
    print("正在提取文本的 tf-idf 特征 ...")

    n_features = 1000

    # 计算文本的tf-idf作为输入NMF模型的特征（feature）.
    # 1max_df=0.95, min_df=2： 使用一些启发式(heuristic)规则预处理去掉一些词语,
    #   删除只在一个文本中出现或者在95%以上的文本中出现的词语
    # max_features=n_features: 预处理后，在剩余的词语里面保留数据集中最常见的n_feature个词语
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                       max_features=n_features,
                                       stop_words='english')
    t0 = time()
    tfidf = tfidf_vectorizer.fit_transform(data_samples)
    print("完成，共耗时 %0.3f 秒." % (time() - t0))
    print("2000篇文本数据集的 bag-of-word 表示:")
    # Fit the NMF model
    n_topics = 10
    print("学习 NMF 分解来拟合 tfidf 特征矩阵, NMF使用%d个话题（topics）..."
          % (n_topics))
    t0 = time()
    nmf = NMF(n_components=n_topics, random_state=1,
              alpha=.1, l1_ratio=.5).fit(tfidf)
    print("完成，共耗时 %0.3f 秒." % (time() - t0))
    # nmf模型对文本的理解保存在nmf.components_参数矩阵中

    print("\n每个话题的代表词语有:")
    n_top_words = 10

    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    print_top_words(nmf, tfidf_feature_names, n_top_words)
    id = [None] * 4
    id[0] = tfidf_feature_names.index('software')
    id[1] = tfidf_feature_names.index('computer')
    id[2] = tfidf_feature_names.index('faith')
    id[3] = tfidf_feature_names.index('bible')

    xs = [None] * 4
    for i in range(4):
        xs[i] = nmf.components_[:, id[i]]
    print("四个单词在话题空间的坐标/表示(representation)/特征(feature)\n")

    print("每个单词是一个%d维的词向量:" % (n_topics))

    print(np.array(xs).T)


if __name__ == "__main__":
    print('开始')
    main()
