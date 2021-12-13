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
learning_rate = 1e-3
epochs = 100
batch_size = 1024
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


def calc_sample_x(transaction, env):
    transaction = np.array(list(map(int, transaction.split(" "))))
    return env.calc_utility_x(transaction)


def build_model(input_dim):
    model = nn.Sequential(
        nn.Linear(input_dim, 512),
        nn.BatchNorm1d(512),
        nn.LeakyReLU(),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.LeakyReLU(),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.LeakyReLU(),
        nn.Linear(128, 1)
    )
    return model


def setup_dataset(feature_path, label_path):
    # load
    X = np.load(feature_path, allow_pickle=True)
    y = np.load(label_path, allow_pickle=True)
    # cast
    X = torch.from_numpy(X).to(device)
    y = torch.from_numpy(y.reshape(-1, 1)).to(device)
    # setup
    dataset = TensorDataset(X, y)
    train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=0.2)
    train_data = Subset(dataset, train_idx)
    test_data = Subset(dataset, test_idx)
    return train_data, test_data


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
    print(f"Test Error: \n Avg RMSE loss: {test_loss:>8f} \n")

    return test_loss


def train(dataset):
    models = []

    kfold = KFold(n_splits=fold, shuffle=True)
    for i, (train_idx, valid_idx) in enumerate(kfold.split(dataset)):
        train_data = Subset(dataset, train_idx)
        train_dataloader = DataLoader(train_data, batch_size)
        valid_data = Subset(dataset, valid_idx)
        valid_dataloader = DataLoader(valid_data, batch_size)

        input_dim = len(dataset[0][0])
        model = build_model(input_dim).to(device)
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        early_stopping = EarlyStopping(patience=patience, verbose=True)
        wandb.watch(model, criterion=loss_fn, idx=i)

        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
            val_loss = test_loop(valid_dataloader, model, loss_fn)
            wandb.log({f"{i}-Fold RMSE train loss": train_loss})
            wandb.log({f"{i}-Fold RMSE val loss": val_loss})
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        model.load_state_dict(torch.load('checkpoint.pt'))
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


def test_sample(test_data, models, sample_n):
    test_dataloader = DataLoader(test_data, 1)
    losses = []
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
            utility = env.calc_utility_bv(X)
            print(f"Predict: {pred_score:>3f}, Utility: {utility}\n")
    print("------------------------------")


if __name__ == "__main__":
    set_seed()
    feature_npy = f"dataset/{DATA}_feature_{DATASET_NUM}.npy"
    label_npy = f"dataset/{DATA}_label_{DATASET_NUM}.npy"
    train_data, test_data = setup_dataset(feature_npy, label_npy)
    models = train(train_data)
    test(test_data, models)
    test_sample(test_data, models, 10)

    for i, model in enumerate(models):
        model_path = f"model/{DATA}_{DATASET_NUM}_{i}.pth"
        torch.save(model.to("cpu").state_dict(), model_path)
