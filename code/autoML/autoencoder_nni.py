import torch
import torch.nn as nn
import torch.optim as optim
import nni
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd

class Autoencoder(nn.Module):
    def __init__(self, hidden_size1, hidden_size2):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(7, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size2, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, 7)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def load_trained_model(model_path):
    model = Autoencoder(4, 3)  # 使用最佳参数创建模型实例
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def encode_data_point(model, data_point):
    with torch.no_grad():
        data_point_tensor = torch.tensor(data_point, dtype=torch.float32).unsqueeze(0)  # 将数据点转换为单批次的张量
        encoded_data = model.encoder(data_point_tensor)
    return encoded_data.numpy()

def main(params,data):
    hidden_size1 = params["hidden_size1"]
    hidden_size2 = params["hidden_size2"]
    learning_rate = params["learning_rate"]

    if hidden_size2 >= hidden_size1:
        raise nni.utils.NNIUserError("hidden_size2 should be less than hidden_size1")

    model = Autoencoder(hidden_size1, hidden_size2)
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_size = int(0.8 * len(data))
    val_size = len(data) - train_size
    train_data, val_data = torch.utils.data.random_split(data, [train_size, val_size])

    batch_size = 32
    # train_loader = DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(TensorDataset(val_data), batch_size=batch_size)

    # 创建 DataLoaders
    train_loader = DataLoader(TensorDataset(train_data.dataset[train_data.indices]), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_data.dataset[val_data.indices]), batch_size=batch_size)


    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            inputs = batch[0]
            outputs = model(inputs)
            loss = criterion(outputs, inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[0]
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        # 报告损失给 NNI
        nni.report_intermediate_result(loss.item())

    # 在训练结束时报告最终损失给 NNI
    nni.report_final_result(val_loss)

def get_criterion(loss_function):
    if loss_function == "mse":
        return nn.MSELoss()
    elif loss_function == "mae":
        return nn.L1Loss()
    elif loss_function == "smooth_l1":
        return nn.SmoothL1Loss()
    else:
        raise ValueError("Invalid loss function")

def train_best_model(params, data):
    hidden_size1 = params["hidden_size1"]
    hidden_size2 = params["hidden_size2"]
    learning_rate = params["learning_rate"]
    loss_function = params["loss_function"]
    batch_size = 32

    model = Autoencoder(hidden_size1, hidden_size2)
    criterion = get_criterion(loss_function)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_size = int(0.8 * len(data))
    val_size = len(data) - train_size
    train_data, val_data = torch.utils.data.random_split(data, [train_size, val_size])

    train_loader = DataLoader(TensorDataset(train_data.dataset[train_data.indices]), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_data.dataset[val_data.indices]), batch_size=batch_size)

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            inputs = batch[0]
            outputs = model(inputs)
            loss = criterion(outputs, inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[0]
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch: {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), "best_autoencoder_model.pth")
    print("Model saved as best_autoencoder_model.pth")

def get_csv_data(path,wanted='1'):
    result=[]
    tot=1
    with open(path) as f:
        context=f.read().split('\n')
        for line in context:
            if tot==1:
                tot=0
                continue
            if line=='':
                continue
            l=line.split(',')
            if wanted!='1':
                if l[-1]!=wanted:
                    continue
            left=[]
            right=[]
            for i in range(7):
                left.append(float(l[i+1]))
                right.append(float(l[i+8]))
            result.append(left)
            result.append(right)
    return np.array(result)



if __name__ == "__main__":
    numpy_data=get_csv_data('../../data/audiogram_concate_withoutNan_class.csv')
    torch_data = torch.from_numpy(numpy_data).float()
    # 归一化操作
    torch_data /= torch.max(torch_data)
    # params = nni.get_next_parameter()
    # main(params, torch_data)
    best_params = {
        "learning_rate": 0.0012017347569035811,
        "hidden_size1": 4,
        "hidden_size2": 3,
        "loss_function": "mse"
    }
    train_best_model(best_params, torch_data)