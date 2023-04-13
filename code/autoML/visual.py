from autoencoder_nni import load_trained_model,encode_data_point,get_csv_data

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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

if __name__ == "__main__":
    Dimension_reduction_tsne()