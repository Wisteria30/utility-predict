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
from model import LengthBasedMLP
from pytorchtools import EarlyStopping
# Predict Utility

fold = 5
learning_rate = 1e-3
epochs = 500
batch_size = 1024
# early stopping patience; how long to wait after last time validation loss improved.
patience = 10
divisions = 3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

wandb.init(project="util-pred", entity="wis30")
wandb.config = {
    "learning_rate": learning_rate,
    "epochs": epochs,
    "batch_size": batch_size,
    "K-fold": fold,
    "Early stopping patience": patience
}


def setup_dataset(X, y, data_split=True):
    # cast
    X = torch.from_numpy(X).to(device)
    y = torch.from_numpy(y.reshape(-1, 1)).to(device)
    # setup
    dataset = TensorDataset(X, y)
    if data_split:
        train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=0.2, shuffle=False)
        train_data = Subset(dataset, train_idx)
        test_data = Subset(dataset, test_idx)
        return train_data, test_data
    return dataset


def train_loop(dataloader, model, loss_fn, optimizers):
    model.train()
    size = len(dataloader.dataset)
    train_loss = None
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        for optimizer in optimizers:
            optimizer.zero_grad()
        loss.backward()
        for optimizer in optimizers:
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
    # print(f"Test Error: \n Avg RMSE loss: {test_loss:>8f} \n")

    return test_loss


def train(dataset, env):
    models = []

    kfold = KFold(n_splits=fold, shuffle=True)
    for i, (train_idx, valid_idx) in enumerate(kfold.split(dataset)):
        train_data = Subset(dataset, train_idx)
        train_dataloader = DataLoader(train_data, batch_size)
        valid_data = Subset(dataset, valid_idx)
        valid_dataloader = DataLoader(valid_data, batch_size)

        input_dim = len(env.items)
        model = LengthBasedMLP(input_dim, divisions, env.tran_length, device)
        loss_fn = nn.MSELoss()
        optimizers = [optim.Adam(model.models[i].parameters(), lr=learning_rate) for i in range(divisions)]
        early_stopping = EarlyStopping(patience=patience, verbose=True)
        # wandb.watch(model, criterion=loss_fn, idx=i)

        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train_loss = train_loop(train_dataloader, model, loss_fn, optimizers)
            val_loss = test_loop(valid_dataloader, model, loss_fn)
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
        losses.append(test_loss)

    loss = np.average(losses)
    print("Test Loss: ", loss)
    wandb.log({"Test Loss": loss})


def test_each_length(test_data, models):
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
        print(f"{i + 1}-Itemset RMSE Loss: {loss}")


def test_sample(test_data, models, sample_n):
    test_dataloader = DataLoader(test_data, 1)
    losses = 0
    loss_fn = nn.MSELoss()
    env = HighUtilityItemsetsMining()

    print("\nSampling")
    print("------------------------------")
    with torch.no_grad():
        for i, (X, y) in enumerate(test_dataloader):
            if i + 1 == sample_n:
                break
            pred_score = 0
            for model in models:
                pred = model(X)
                pred_score += pred.item()

            pred_score /= len(models)
            X = X[0].to("cpu").detach().numpy().copy()
            itemsets, utility = env.calc_utility_bv(X)
            print(f"{itemsets} => Predict: {pred_score:>3f}, Utility: {utility}")
    print("------------------------------")


def main():
    set_seed()
    feature_npy = f"dataset/{DATA}_feature_{DATASET_NUM}.npy"
    label_npy = f"dataset/{DATA}_label_{DATASET_NUM}.npy"
    # load
    X = np.load(feature_npy, allow_pickle=True)
    y = np.load(label_npy, allow_pickle=True)
    train_data, test_data = setup_dataset(X, y)
    env = HighUtilityItemsetsMining()
    models = train(train_data, env)
    test(test_data, models)
    test_sample(test_data, models, 10)
    test_each_length(test_data, models)

    for i, model in enumerate(models):
        model_path = f"model/length/{DATA}_{DATASET_NUM}_{i}.pth"
        torch.save(model.to("cpu").state_dict(), model_path)


def predict():
    set_seed()
    feature_npy = f"dataset/{DATA}_feature_{DATASET_NUM}.npy"
    label_npy = f"dataset/{DATA}_label_{DATASET_NUM}.npy"
    # load
    X = np.load(feature_npy, allow_pickle=True)
    y = np.load(label_npy, allow_pickle=True)
    _, test_data = setup_dataset(X, y)
    env = HighUtilityItemsetsMining()
    input_dim = len(env.items)

    models = []
    for i in range(5):
        path = f"model/length/{DATA}_{DATASET_NUM}_{i}.pth"
        model = LengthBasedMLP(input_dim, divisions, env.tran_length, device)
        model.load_state_dict(torch.load(path))
        model.eval()
        models.append(model)

    test(test_data, models)
    test_sample(test_data, models, 10)
    test_each_length(test_data, models)


if __name__ == "__main__":
    predict()
