import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# t-SNE is used to visualize the distribution of learned features in a 2D embedding space. Each point represents a sample, and nearby points indicate similar feature representations in the original high-dimensional space.
# “We visualize high-dimensional texture features using t-SNE. The resulting 2D coordinates do not correspond to physical meaning but preserve local neighborhood relationships in the embedding space.”“The axes correspond to arbitrary t-SNE embedding dimensions without physical meaning.”
# “The coordinates are unitless and have no direct semantic interpretation.”
# tsne输出低维嵌入坐标（arbitrary coordinate system） neighbor similarity in high-D≈neighbor similarity in 2D
# 所以 x / y 的意义：没有物理含义axis 方向可以旋转可以翻转可以缩放 甚至可以 non-linear warp   点与点之间的相对距离（局部）有意义
def tsne(features,labels,num_cls):
    # 对特征做标准化（建议）
    features_std = StandardScaler().fit_transform(features)

    # 进行 t-SNE 降维到 2D
    # perplexity: effective number of neighbors 小perplexity（5 ~ 20）强调局部结构，类内小簇更紧更容易“撕裂”大类类间可能更碎 大 perplexity（50 ~ 100+）强调全局结构cluster 更“平滑”类间边界可能模糊小类可能被压没
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features_std)

    # 可视化不同类别的聚类分布
    plt.figure(1,figsize=(8, 6))
    # 5个类别
    for i in range(num_cls):
        plt.scatter(features_2d[labels == i, 0], features_2d[labels == i, 1], label=f'Class {i}', alpha=0.6)
    plt.title("t-SNE visualization of texture features")
    plt.legend()
    plt.grid(True)
    plt.show()

def pca(features,labels):
    features_std = StandardScaler().fit_transform(features)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(features_std)

    plt.figure(2,figsize=(8,6))
    for i in range(4):
        plt.scatter(X_pca[labels==i, 0], X_pca[labels==i, 1], label=f"Class {i}", alpha=0.6)

    plt.title("PCA Visualization of Texture Features")
    plt.legend()
    plt.grid(True)
    plt.show()