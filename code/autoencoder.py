import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# 假设你的数据存储在 data 中
data = np.array(...) # 将省略号替换为你的实际数据

# 标准化数据
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# 转换为 PyTorch 张量
data_tensor = torch.FloatTensor(data_scaled)

# 创建数据加载器
batch_size = 64
dataset = TensorDataset(data_tensor, data_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 设置自编码器参数
input_dim = data_scaled.shape[1]
encoding_dim = 3

# 定义自编码器模型
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 实例化模型
autoencoder = Autoencoder(input_dim, encoding_dim)

# 设置优化器和损失函数
optimizer = optim.Adam(autoencoder.parameters())
loss_function = nn.MSELoss()

# 训练模型
epochs = 100
for epoch in range(epochs):
    for batch_features, _ in dataloader:
        optimizer.zero_grad()
        outputs = autoencoder(batch_features)
        loss = loss_function(outputs, batch_features)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# 提取编码器部分
encoder = autoencoder.encoder

# 对数据进行编码
encoded_data = encoder(torch.FloatTensor(data_scaled)).detach().numpy()

# 输出编码后的数据
print("编码后的数据:", encoded_data)


# 假设 encoded_data 是自编码器编码后的数据
encoded_data = np.array(...)  # 将省略号替换为实际的编码后数据

# 使用 K-means 进行聚类
n_clusters = 3  # 假设我们想要找到 3 个类别
kmeans = KMeans(n_clusters=n_clusters)
cluster_labels = kmeans.fit_predict(encoded_data)

# 可视化聚类结果（假设编码后的数据是二维或三维的）
if encoded_data.shape[1] == 2:
    plt.scatter(encoded_data[:, 0], encoded_data[:, 1], c=cluster_labels)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('2D Visualization of Clustering Results')
    plt.show()
elif encoded_data.shape[1] == 3:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(encoded_data[:, 0], encoded_data[:, 1], encoded_data[:, 2], c=cluster_labels)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    plt.title('3D Visualization of Clustering Results')
    plt.show()


# 假设 encoded_data 是自编码器编码后的数据
encoded_data = np.array(...)  # 将省略号替换为实际的编码后数据

# 使用 K-means 进行聚类
n_clusters = 3  # 假设我们想要找到 3 个类别
kmeans = KMeans(n_clusters=n_clusters)
cluster_labels = kmeans.fit_predict(encoded_data)

# 可视化聚类结果（假设编码后的数据是二维或三维的）
if encoded_data.shape[1] == 2:
    plt.scatter(encoded_data[:, 0], encoded_data[:, 1], c=cluster_labels)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('2D Visualization of Clustering Results')
    plt.show()
elif encoded_data.shape[1] == 3:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(encoded_data[:, 0], encoded_data[:, 1], encoded_data[:, 2], c=cluster_labels)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    plt.title('3D Visualization of Clustering Results')
    plt.show()

# 对每个类别的数据进行统计分析
for i in range(n_clusters):
    print(f"Cluster {i + 1}:")
    cluster_data = encoded_data[cluster_labels == i]

    # 计算均值和标准差
    mean = np.mean(cluster_data, axis=0)
    std = np.std(cluster_data, axis=0)
    print(f"Mean: {mean}")
    print(f"Standard Deviation: {std}")

    # 可视化原始数据中的对应样本
    original_cluster_data = data[cluster_labels == i]

    # 使用 PCA 将原始数据降至二维
    pca = PCA(n_components=2)
    original_cluster_data_2d = pca.fit_transform(StandardScaler().fit_transform(original_cluster_data))

    plt.scatter(original_cluster_data_2d[:, 0], original_cluster_data_2d[:, 1])
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(f'2D Visualization of Original Data in Cluster {i + 1}')
    plt.show()
