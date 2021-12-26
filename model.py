import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        hidden_1 = 512
        hidden_2 = 256
        hidden_3 = 128
        output = 1

        self.fc1 = nn.Linear(input_dim, hidden_1)
        self.bn1 = nn.BatchNorm1d(hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.bn2 = nn.BatchNorm1d(hidden_2)
        self.fc3 = nn.Linear(hidden_2, hidden_3)
        self.bn3 = nn.BatchNorm1d(hidden_3)
        self.fc4 = nn.Linear(hidden_3, output)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = F.leaky_relu(self.bn3(self.fc3(x)))
        return self.fc4(x)


class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        hidden_1 = 512
        hidden_2 = 256
        hidden_3 = 128
        output = 1

        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.fc1 = nn.Linear(embedding_dim, hidden_1)
        self.bn1 = nn.BatchNorm1d(hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.bn2 = nn.BatchNorm1d(hidden_2)
        self.fc3 = nn.Linear(hidden_2, hidden_3)
        self.bn3 = nn.BatchNorm1d(hidden_3)
        self.fc4 = nn.Linear(hidden_3, output)

    def forward(self, x):
        x = self.embeddings(x)
        # 0-paddingの影響で平均がおかしくなるので、それぞれ有効なベルトルの数をカウントして自前で平均計算
        mask = torch.sum(x, 2) != 0
        # broadcast
        x = torch.sum(x, 1) / mask.sum(dim=1)[:, None]
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = F.leaky_relu(self.bn3(self.fc3(x)))
        return self.fc4(x)


class LengthBasedMLP(nn.Module):
    def __init__(self, input_dim, divisions, length, device):
        super().__init__()
        self.divisions = divisions
        self.length = length
        self.device = device
        self.models = nn.ModuleList([MLP(input_dim).to(device) for _ in range(divisions)])

    def forward(self, x):
        pred = torch.zeros(x.size()[0], 1).to(self.device)
        # 3分割:37, 
        # 0 < x <= 12(1/3)
        # 12 < x <= 24(2/3)
        # 24 < x <= 37
        for i in range(self.divisions):
            left = i / self.divisions * self.length < torch.sum(x, 1)
            right = torch.sum(x, 1) <= (i + 1) / self.divisions * self.length
            scope = left & right
            index = torch.where(scope)[0]

            x_divisions = x[scope]
            x_divisions = self.models[i](x_divisions)
            for j, idx in enumerate(index):
                pred[idx] = x_divisions[j]

        return pred
