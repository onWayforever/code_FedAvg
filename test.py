# import numpy as np
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt

# # 生成12个二维数据点
# data = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11],
#                  [8, 2], [10, 2], [9, 3], [3, 4], [3, 3.5], [7, 7]])

# # 使用KMeans将数据点分为4个簇
# kmeans = KMeans(n_clusters=4)
# kmeans.fit(data)

# # 获取聚类中心和标签
# centroids = kmeans.cluster_centers_
# labels = kmeans.labels_

# # 打印聚类中心和标签
# print("Centroids:\n", centroids)
# print("Labels:\n", labels)

# # 可视化结果
# colors = ['r', 'g', 'b', 'y']
# for i in range(len(data)):
#     plt.scatter(data[i][0], data[i][1], c=colors[labels[i]], s=100)
    
# plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=150, linewidths=5, zorder=10, c='black')
# plt.show()


# def is_length_consistent(lst):
#     if not lst:
#         return True  # 空列表被视为长度一致
    
#     first_length = len(lst[0])
#     return all(len(item) == first_length for item in lst)

# # 示例列表
# lst1 = ["apple", "banana", "cherry"]
# lst2 = ["dog", "cat", "hot"]

# print(is_length_consistent(lst1))  # 输出: False
# print(is_length_consistent(lst2))  # 输出: True

import random

# 假设我们有 10 个客户端
clients = list(range(10))  # [0, 1, 2, ..., 9]

# 洗牌
random.shuffle(clients)  # 例如：[3, 1, 7, 2, 9, 5, 4, 6, 0, 8]

# 假设我们想要创建 3 个组
num_groups = 3

# 分块
groups = [clients[i::num_groups] for i in range(num_groups)]

for i, group in enumerate(groups, start=1):
    print(f"Group {i}: {group}")
