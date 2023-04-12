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

def main(params,data):
    hidden_size1 = params["hidden_size1"]
    hidden_size2 = params["hidden_size2"]
    learning_rate = params["learning_rate"]
    loss_function = params["loss_function"]

    model = Autoencoder(hidden_size1, hidden_size2)

    # 根据参数选择损失函数
    if loss_function == "l1":
        criterion = nn.L1Loss()
    elif loss_function == "mse":
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss_function: {loss_function}")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_size = int(0.8 * len(data))
    val_size = len(data) - train_size
    train_data, val_data = torch.utils.data.random_split(data, [train_size, val_size])

    batch_size = 32
    train_loader = DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_data), batch_size=batch_size)

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

def get_csv_data(path):
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
    params = nni.get_next_parameter()
    main(params, torch_data)
