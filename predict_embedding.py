import numpy as np
import torch
from torch.multiprocessing import Pool, set_start_method
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.model_selection import KFold, train_test_split
import wandb

# KDE
from scipy.stats import gaussian_kde
import matplotlib
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from config import set_seed, DATASET_NUM, DATA
from hui_env import HighUtilityItemsetsMining
from loss import L1RelativeLoss
from model import EmbeddingModel
from pytorchtools import EarlyStopping
# Predict Utility

fold = 5
learning_rate = 0.05
# learning_rate = 1e4
epochs = 1000
# batch_size = 1024
batch_size = 64
embedding_dim = 128
# early stopping patience; how long to wait after last time validation loss improved.
patience = 10
weight = 1e4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# wandb.init(project="util-pred", entity="wis30")
# wandb.config = {
#     "learning_rate": learning_rate,
#     "epochs": epochs,
#     "batch_size": batch_size,
#     "K-fold": fold,
#     "Early stopping patience": patience
# }

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

    def transform_loss(self, loss):
        return loss * self.std



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
    print("setup dataset")
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
        train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=0.2, shuffle=False)
        train_data = Subset(dataset, train_idx)
        test_data = Subset(dataset, test_idx)
        return train_data, test_data, normalization
    print("Done")
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
            # loss = np.sqrt(loss)
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


def train(dataset, env, normalization):
    models = []

    kfold = KFold(n_splits=fold, shuffle=True)
    for i, (train_idx, valid_idx) in enumerate(kfold.split(dataset)):
        train_data = Subset(dataset, train_idx)
        train_dataloader = DataLoader(train_data, batch_size)
        valid_data = Subset(dataset, valid_idx)
        valid_dataloader = DataLoader(valid_data, batch_size)

        vocab_size = len(env.items) + 1
        model = EmbeddingModel(vocab_size, embedding_dim).to(device)
        loss_fn = L1RelativeLoss(normalization.mean, weight)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        early_stopping = EarlyStopping(patience=patience, verbose=True)
        # wandb.watch(model, criterion=loss_fn, idx=i)

        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
            val_loss = test_loop(valid_dataloader, model, loss_fn)
            print(f"Validation Error: \n Avg loss: {val_loss:>8f} \n")
            # wandb.log({f"{i}-Fold RMSE train loss": train_loss})
            # wandb.log({f"{i}-Fold RMSE val loss": val_loss})
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


def model_parallel(inputs):
    n, train_dataloader, valid_dataloader, env, normalization = inputs
    vocab_size = len(env.items) + 1
    model = EmbeddingModel(vocab_size, embedding_dim).to(device)
    loss_fn = L1RelativeLoss(normalization.mean, weight)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
        val_loss = test_loop(valid_dataloader, model, loss_fn)
        print(f"{n}-Fold Validation Error: \n Avg loss: {val_loss:>8f} \n")
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    model.load_state_dict(torch.load('checkpoint.pt'))
    model.eval()
    print("Fold Done!")
    return model


def train_parallel(dataset, env, normalization):
    kfold = KFold(n_splits=fold, shuffle=True)
    values = []
    for i, (train_idx, valid_idx) in enumerate(kfold.split(dataset)):
        train_data = Subset(dataset, train_idx)
        train_dataloader = DataLoader(train_data, batch_size)
        valid_data = Subset(dataset, valid_idx)
        valid_dataloader = DataLoader(valid_data, batch_size)

        values.append((i, train_dataloader, valid_dataloader, env, normalization))

    p = Pool(fold)
    models = p.map(model_parallel, values)
    print("Train Done!")
    return models

def test(test_data, models, normalization):
    test_dataloader = DataLoader(test_data, batch_size)
    losses = []
    loss_fn = nn.MSELoss()
    for model in models:
        test_loss = test_loop(test_dataloader, model, loss_fn)
        print(f"Test Error: \n Avg RMSE loss: {test_loss:>8f} \n")
        losses.append(test_loss)

    loss = np.average(losses)
    loss_utility = normalization.transform_loss(loss)
    print(f"Test RMSE Loss: {loss:>8f}({loss_utility:>8f})")
    # wandb.log({"Test Loss": loss})


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
        loss_utility = normalization.transform_loss(loss)
        print(f"{i + 1}-Itemset RMSE Loss: {loss:>8f}({loss_utility:>8f})")


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
    models = train_parallel(train_data, env, normalization)
    # models = train(train_data, env, normalization)
    test(test_data, models, normalization)
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

    env = HighUtilityItemsetsMining()
    vocab_size = len(env.items) + 1

    models = []
    for i in range(5):
        path = f"model/embedding/{DATA}_{DATASET_NUM}_{i}.pth"
        model = EmbeddingModel(vocab_size, embedding_dim).to(device)
        model.load_state_dict(torch.load(path))
        model.eval()
        models.append(model)

    test(test_data, models, normalization)
    test_each_length(test_data, models, normalization)


def get_error(dataloader, model):
    errors = np.array([])
    model.eval()
    size = len(dataloader.dataset)
    print("create Error")
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        error = y - pred
        error = error.squeeze().to("cpu").detach().numpy().copy()
        errors = np.concatenate([errors, error])

    return errors


def kde(dataset, env):
    plt.style.use('ggplot')

    dataloader = DataLoader(dataset, 1024)
    vocab_size = len(env.items) + 1
    model = EmbeddingModel(vocab_size, embedding_dim).to(device)

    errors = get_error(dataloader, model)

    # x軸をGridするためのデータも生成
    x_grid = np.linspace(min(errors), max(errors), num=100)
    # データを正規化したヒストグラムを表示する用
    weights = np.ones_like(errors)/float(len(errors))

    kde_model = gaussian_kde(errors)
    y = kde_model(x_grid)

    fig = plt.figure(figsize=(14,7))
    plt.plot(x_grid, y)
    plt.hist(errors, alpha=0.3, bins=20, weights=weights)

    fig.savefig("errors.png")
    print("graph save!")


def error_kde():
    set_seed()
    feature_npy = f"dataset/{DATA}_feature_{DATASET_NUM}.npy"
    label_npy = f"dataset/{DATA}_label_{DATASET_NUM}.npy"
    # load
    X = np.load(feature_npy, allow_pickle=True)
    y = np.load(label_npy, allow_pickle=True)
    env = HighUtilityItemsetsMining()
    train_data, test_data, normalization = setup_dataset(X, y)
    kde(test_data, env)


if __name__ == "__main__":
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        print("RuntimeError")
        pass
    main()
