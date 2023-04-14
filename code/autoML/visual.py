from autoencoder_nni import load_trained_model,encode_data_point,get_csv_data

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

def Dimension_reduction_tsne():
    numpy_data=get_csv_data('../../data/audiogram_concate_withoutNan_class.csv','None')
    list_data=numpy_data.tolist()
    # 加载模型
    trained_model = load_trained_model("best_autoencoder_model.pth")
    encoded_data = encode_data_point(trained_model, list_data)
    encoded_data = encoded_data[0]

    # 使用 t-SNE 进行降维
    tsne = TSNE(n_components=2, random_state=42)
    print("开始降维")
    encoded_data_2d = tsne.fit_transform(encoded_data)
    print("降维结束")

    # 可视化降维后的数据
    plt.scatter(encoded_data_2d[:, 0], encoded_data_2d[:, 1])
    plt.title("t-SNE visualization of encoded data")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()
    plt.savefig("./t-SNE visualization of encoded data.png")

def Dimension_reducation_pca():
    numpy_data=get_csv_data('../../data/audiogram_concate_withoutNan_class.csv')
    list_data=numpy_data.tolist()
    # 加载模型
    trained_model = load_trained_model("best_autoencoder_model.pth")
    encoded_data = encode_data_point(trained_model, list_data)
    encoded_data = encoded_data[0]

    # 使用 PCA 进行降维
    pca = PCA(n_components=2, random_state=42)
    print("开始降维")
    encoded_data_2d = pca.fit_transform(encoded_data)
    print("降维结束")


    # 输出解释方差比，表示各主成分解释的原始数据方差的比例
    explained_variance_ratio = pca.explained_variance_ratio_
    print("解释方差比:", explained_variance_ratio)

    # 输出累计解释方差比，表示前 N 个主成分解释的原始数据方差的累计比例
    cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)
    print("累计解释方差比:", cumulative_explained_variance_ratio)

    # 可视化降维后的数据
    plt.scatter(encoded_data_2d[:, 0], encoded_data_2d[:, 1])
    plt.title("PCA visualization of encoded data")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()
    plt.savefig("./PCA visualization of encoded data.png")

def visual_encoder_data():
    numpy_data=get_csv_data('../../data/audiogram_concate_withoutNan_class.csv','None')
    list_data=numpy_data.tolist()
    # 加载模型
    trained_model = load_trained_model("best_autoencoder_model.pth")
    encoded_data = encode_data_point(trained_model, list_data)
    encoded_data = encoded_data[0]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(encoded_data[:, 0], encoded_data[:, 1], encoded_data[:, 2] )
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    plt.title('3D Visualization of encoder feature')
    plt.show()
    plt.savefig("./visualization of encoded data.png")

def k_means():
    numpy_data=get_csv_data('../../data/audiogram_concate_withoutNan_class.csv','None')
    list_data=numpy_data.tolist()
    # 加载模型
    trained_model = load_trained_model("best_autoencoder_model.pth")
    encoded_data = encode_data_point(trained_model, list_data)
    encoded_data = encoded_data[0]


    # 使用肘部法则选择合适的聚类数目
    max_clusters = 2000
    sse = []
    silhouette_coefficients = []

    for k in range(2, max_clusters + 1):
        print("开始聚类数目为", k)
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(encoded_data)
        sse.append(kmeans.inertia_)
        silhouette_coefficients.append(silhouette_score(encoded_data, kmeans.labels_))
    
    with open("./sse.txt", "w") as f:
        f.write(str(sse))
    
    with open("./silhouette_coefficients.txt", "w") as f:
        f.write(str(silhouette_coefficients))

    # 绘制肘部法则图
    plt.plot(range(2, max_clusters + 1), sse)
    plt.xlabel('Number of Clusters')
    plt.ylabel('SSE')
    plt.title('Elbow Method')
    plt.savefig("./Elbow Method.png")
    plt.show()
    plt.close()

    # 绘制轮廓系数图
    plt.plot(range(2, max_clusters + 1), silhouette_coefficients)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Coefficient')
    plt.title('Silhouette Coefficient')
    plt.savefig("./Silhouette Coefficient.png")
    plt.show()



def plot_k_means_result(n_cluster):
    # plot the result of k-means
    # use the t-sne result as the input
    numpy_data=get_csv_data('../../data/audiogram_concate_withoutNan_class.csv','None')
    list_data=numpy_data.tolist()
    # 加载模型
    trained_model = load_trained_model("best_autoencoder_model.pth")
    encoded_data = encode_data_point(trained_model, list_data)
    encoded_data = encoded_data[0]

    # 使用 t-SNE 进行降维
    # tsne = TSNE(n_components=2, random_state=42)
    # print("开始降维")
    # encoded_data_2d = tsne.fit_transform(encoded_data)
    # print("降维结束")

    # 使用 K-Means 进行聚类
    # kmeans = KMeans(n_clusters=n_cluster, random_state=42)
    kmeans = KMeans(n_clusters=n_cluster, random_state=42).fit(encoded_data)
    print("聚类结束")
    '''
    cluster_labels = kmeans.fit_predict(encoded_data)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(encoded_data[:, 0], encoded_data[:, 1], encoded_data[:, 2],c=cluster_labels, cmap='rainbow')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    plt.title('3D Visualization of encoder feature')
    plt.show()
    plt.savefig("./visualization of encoded data.png")
    '''
    # 找到距离每个类质心最近的数据点
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    closest_indices = []
    for i in range(len(centers)):
        center = centers[i]
        distances = [((encoded_data[j] - center) ** 2).sum() for j in range(len(encoded_data)) if labels[j] == i]
        closest_index = [j for j in range(len(encoded_data)) if labels[j] == i][distances.index(min(distances))]
        closest_indices.append(closest_index)

    # 绘制并保存原始数据图
    if not os.path.exists("clustered_data"):
        os.mkdir("clustered_data")

    x = [500, 1000, 2000, 3000, 4000, 6000, 8000]
    for i, index in enumerate(closest_indices):
        plt.figure()
        # 绘制折线图
        plt.plot(x, list_data[index], color='blue', linewidth=1 )
        # 绘制散点图
        plt.plot(x, list_data[index], 'o', markersize=10)
        plt.xticks(x, ['500', '1kHz', '2kHz', '3kHz', '4kHz', '6kHz', '8kHz'])
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Threshold (dB)')
        plt.savefig("clustered_data/cluster_{}.png".format(i+1))
        plt.close() 

from PIL import Image

def combine_fig():
    # 定义图片数量和行列数
    num_images = 24
    num_rows = 4
    num_cols = 6

    # 计算每张图片的大小
    width, height = Image.open('./clustered_data/cluster_1.png').size

    # 创建一个新的大图
    new_im = Image.new('RGB', (num_cols * width, num_rows * height))

    # 循环遍历每张小图并将其粘贴到大图上
    for i in range(num_images):
        im = Image.open('./clustered_data/cluster_{}.png'.format(i+1))
        row = i // num_cols
        col = i % num_cols
        new_im.paste(im, (col*width, row*height))

    # 保存拼接后的图片
    new_im.save('combined.png')


if __name__ == "__main__":
    # Dimension_reduction_tsne()
    # Dimension_reducation_pca()
    # visual_encoder_data()
    # k_means()
    # plot_k_means_result(25)
    combine_fig()