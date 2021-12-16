import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.model_selection import KFold, train_test_split
import wandb

from config import set_seed, DATASET_NUM, DATA
from hui_env import HighUtilityItemsetsMining
from pytorchtools import EarlyStopping
# Predict Utility

fold = 5
learning_rate = 0.01
epochs = 500
batch_size = 128
embedding_dim = 128
# early stopping patience; how long to wait after last time validation loss improved.
patience = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

wandb.init(project="util-pred", entity="wis30")
wandb.config = {
    "learning_rate": learning_rate,
    "epochs": epochs,
    "batch_size": batch_size,
    "K-fold": fold,
    "Early stopping patience": patience
}


class Model(nn.Module):
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
        x = torch.mean(x, 1)
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = F.leaky_relu(self.bn3(self.fc3(x)))
        return self.fc4(x)


class Normalization:
    def __init__(self):
        self.mean = None
        self.std = None

    def get_z(self, x, ddof=1):
        self.mean = np.average(x)
        self.std = np.std(x, ddof=ddof)
        z = (x - self.mean) / self.std
        return z
    
    def get_x(self, z):
        x = (z * self.std) + self.mean
        return x


def convert_embedding2bv(x):
    # 修正後注意
    bv = np.zeros(len(x))
    for i in x:
        if i == 0:
            continue
        # 0-paddingの影響でアイテムは1からアイテム数まで
        bv[i - 1] = 1.
    return bv


def setup_dataset(X, y, data_split=True):
    # Embedding Layerに突っ込むので、one-hotではなくindexのリストをintで突っ込む
    vocab_data = [[] for _ in range(len(X))]
    for i, j in zip(*np.where(X > 0)):
        vocab_data[i].append(j + 1)

    # 0 padding
    for i in range(len(vocab_data)):
        padding_length = len(X[0]) - len(vocab_data[i])
        vocab_data[i] = np.pad(vocab_data[i], [0, padding_length], 'constant')

    vocab_data = np.array(vocab_data, dtype=int)
    # 標準化
    normalization = Normalization()
    y = normalization.get_z(y)
    # cast
    X = torch.from_numpy(vocab_data).to(torch.long).to(device)
    y = torch.from_numpy(y.reshape(-1, 1)).to(device)
    # setup
    dataset = TensorDataset(X, y)
    if data_split:
        train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=0.2)
        train_data = Subset(dataset, train_idx)
        test_data = Subset(dataset, test_idx)
        return train_data, test_data, normalization
    return dataset, normalization


def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)
    train_loss = None
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            loss = np.sqrt(loss)
            train_loss = loss
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return train_loss


def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    test_loss = np.sqrt(test_loss)

    return test_loss


def train(dataset, env):
    models = []

    kfold = KFold(n_splits=fold, shuffle=True)
    for i, (train_idx, valid_idx) in enumerate(kfold.split(dataset)):
        train_data = Subset(dataset, train_idx)
        train_dataloader = DataLoader(train_data, batch_size)
        valid_data = Subset(dataset, valid_idx)
        valid_dataloader = DataLoader(valid_data, batch_size)

        vocab_size = len(env.items) + 1
        model = Model(vocab_size, embedding_dim).to(device)
        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        early_stopping = EarlyStopping(patience=patience, verbose=True)
        wandb.watch(model, criterion=loss_fn, idx=i)

        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
            val_loss = test_loop(valid_dataloader, model, loss_fn)
            print(f"Validation Error: \n Avg RMSE loss: {val_loss:>8f} \n")
            wandb.log({f"{i}-Fold RMSE train loss": train_loss})
            wandb.log({f"{i}-Fold RMSE val loss": val_loss})
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        model.load_state_dict(torch.load('checkpoint.pt'))
        model.eval()
        models.append(model)
        print("Fold Done!")
    print("Train Done!")

    return models


def test(test_data, models):
    test_dataloader = DataLoader(test_data, batch_size)
    losses = []
    loss_fn = nn.MSELoss()
    for model in models:
        test_loss = test_loop(test_dataloader, model, loss_fn)
        print(f"Test Error: \n Avg RMSE loss: {test_loss:>8f} \n")
        losses.append(test_loss)

    loss = np.average(losses)
    print("Test Loss: ", loss)
    wandb.log({"Test Loss": loss})


def test_each_length(test_data, models, normalization):
    itemsets_length_idx = [[] for _ in range(len(test_data[0][0]))]
    # アイテムセットの長さでindex分ける（トランザクションの長さがわからないので、アイテムの長さでloop）
    for i, data in enumerate(test_data):
        itemset_length = len(data[0][data[0] > 0])
        itemsets_length_idx[itemset_length - 1].append(i)

    loss_fn = nn.MSELoss()
    print("Test")

    for i, v in enumerate(itemsets_length_idx):
        if len(v) == 0:
            continue
        x_length_data = Subset(test_data, v)
        dataloader = DataLoader(x_length_data, batch_size)
        loss = 0
        for model in models:
            loss += test_loop(dataloader, model, loss_fn)
        loss /= len(models)
        loss_x = normalization.get_x(loss)
        print(f"{i + 1}-Itemset RMSE Loss: {loss:>8f}({loss_x})")


def test_sample(test_data, models, normalization, sample_n):
    test_dataloader = DataLoader(test_data, 1)
    env = HighUtilityItemsetsMining()

    print("\nSampling")
    print("------------------------------")
    with torch.no_grad():
        for i, (X, y) in enumerate(test_dataloader):
            if i + 1 == sample_n:
                break
            pred_utility = 0
            for model in models:
                pred = model(X)
                pred_utility += pred.item()

            pred_utility /= len(models)
            pred_utility = normalization.get_x(pred_utility)
            X = X[0].to("cpu").detach().numpy().copy()
            X = convert_embedding2bv(X)
            itemsets, utility = env.calc_utility_bv(X)
            print(f"{itemsets} => Predict: {pred_utility:>3f}, Utility: {utility}")
    print("------------------------------")


def main():
    set_seed()
    feature_npy = f"dataset/{DATA}_feature_{DATASET_NUM}.npy"
    label_npy = f"dataset/{DATA}_label_{DATASET_NUM}.npy"
    # load
    X = np.load(feature_npy, allow_pickle=True)
    y = np.load(label_npy, allow_pickle=True)
    env = HighUtilityItemsetsMining()
    train_data, test_data, normalization = setup_dataset(X, y)
    models = train(train_data, env)
    test(test_data, models)
    test_sample(test_data, models, normalization, 10)
    test_each_length(test_data, models, normalization)

    for i, model in enumerate(models):
        model_path = f"model/embedding/{DATA}_{DATASET_NUM}_{i}.pth"
        torch.save(model.to("cpu").state_dict(), model_path)


def predict():
    set_seed()
    feature_npy = f"dataset/{DATA}_feature_{DATASET_NUM}.npy"
    label_npy = f"dataset/{DATA}_label_{DATASET_NUM}.npy"
    # load
    X = np.load(feature_npy, allow_pickle=True)
    y = np.load(label_npy, allow_pickle=True)
    _, test_data, normalization = setup_dataset(X, y)
    items_length = len(test_data[0][0])

    models = []
    for i in range(5):
        path = f"model/embedding/{DATA}_{DATASET_NUM}_{i}.pth"
        model = build_model(items_length).to(device)
        model.load_state_dict(torch.load(path))
        model.eval()
        models.append(model)

    test_each_length(test_data, models)


if __name__ == "__main__":
    main()
