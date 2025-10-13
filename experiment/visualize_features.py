import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
def tsne(features,labels):
    # 对特征做标准化（建议）
    features_std = StandardScaler().fit_transform(features)

    # 进行 t-SNE 降维到 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features_std)

    # 可视化不同类别的聚类分布
    plt.figure(1,figsize=(8, 6))
    # 5个类别
    for i in range(4):
        plt.scatter(features_2d[labels == i, 0], features_2d[labels == i, 1], label=f'Class {i}', alpha=0.6)
    plt.title("t-SNE visualization of texture features")
    plt.legend()
    plt.grid(True)
    # plt.show()

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