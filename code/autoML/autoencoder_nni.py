import torch
import torch.nn as nn
import torch.optim as optim
import nni

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

def main(params):
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

    data = torch.tensor([[1, 2, 3, 4, 5, 6, 7]], dtype=torch.float32)  # 示例数据，请用实际数据替换
    data /= torch.max(data)  # 归一化数据

    num_epochs = 100
    for epoch in range(num_epochs):
        outputs = model(data)
        loss = criterion(outputs, data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 报告损失给 NNI
        nni.report_intermediate_result(loss.item())

    # 在训练结束时报告最终损失给 NNI
    nni.report_final_result(loss.item())

if __name__ == "__main__":
    params = nni.get_next_parameter()
    main(params)
