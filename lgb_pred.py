import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, train_test_split
import wandb
from wandb.lightgbm import wandb_callback

from config import set_seed, DATASET_NUM, DATA
from hui_env import HighUtilityItemsetsMining
from model import LengthBasedMLP
from pytorchtools import EarlyStopping
# Predict Utility

fold = 5
learning_rate = 1e-3
epochs = 500
verbose_eval = -1
batch_size = 1024
# early stopping patience; how long to wait after last time validation loss improved.
patience = 10
divisions = 3

wandb.init(project="util-pred", entity="wis30")
wandb.config = {
    "learning_rate": learning_rate,
    "epochs": epochs,
    "batch_size": batch_size,
    "K-fold": fold,
    "Early stopping patience": patience
}

params = {
    'objective': 'regression',
    'verbose': -1,
}


def setup_dataset(X, y, data_split=True):
    if data_split:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False)
        return X_train, X_test, y_train, y_test
    return X, y


def train(X, y):
    valid_scores = []
    models = []
    kfold = KFold(n_splits=fold, shuffle=True)
    for i, (train_idx, valid_idx) in enumerate(kfold.split(X)):
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_valid, y_valid)

        model = lgb.train(
            params,
            lgb_train,
            valid_sets=lgb_eval,
            num_boost_round=epochs,
            early_stopping_rounds=patience,
            callbacks=[wandb_callback()],
            verbose_eval=verbose_eval
        )

        y_valid_pred = model.predict(X_valid)
        score = np.sqrt(mean_squared_error(y_valid, y_valid_pred))
        print(f'fold {i} RMSE: {score}')
        wandb.log({f"{i}-Fold RMSE val loss": score})
        valid_scores.append(score)

        models.append(model)
        print("Fold Done!")

    cv_score = np.mean(valid_scores)
    print(f'CV score: {cv_score}')
    print("Train Done!")
    return models


def test(X, y, models):
    loss = 0

    for model in models:
        y_pred = model.predict(X, num_iteration=model.best_iteration)
        loss += np.sqrt(mean_squared_error(y, y_pred))
    loss /= len(models)

    print("Test Loss: ", loss)
    wandb.log({"Test Loss": loss})


def test_each_length(X, y, models):
    itemsets_length_idx = [[] for _ in range(len(X[0]))]
    # アイテムセットの長さでindex分ける（トランザクションの長さがわからないので、アイテムの長さでloop）
    for i, x in enumerate(X):
        itemset_length = len(x[x > 0])
        itemsets_length_idx[itemset_length - 1].append(i)

    print("Test")
    for i, v in enumerate(itemsets_length_idx):
        loss = 0
        if len(v) == 0:
            continue
        for model in models:
            y_pred = model.predict(X[v], num_iteration=model.best_iteration)
            loss += np.sqrt(mean_squared_error(y[v], y_pred))
        loss /= len(models)
        print(f"{i + 1}-Itemset RMSE Loss: {loss}")


def test_sample(X, y, models, sample_n):
    env = HighUtilityItemsetsMining()

    print("\nSampling")
    print("------------------------------")
    for i, x in enumerate(X):
        if i + 1 == sample_n:
            break
        pred_score = 0
        for model in models:
            y_pred = model.predict(x.reshape(1, -1), num_iteration=model.best_iteration)
            pred_score += y_pred[0]

        pred_score /= len(models)
        itemsets, utility = env.calc_utility_bv(x)
        print(f"{itemsets} => Predict: {pred_score:>3f}, Utility: {utility}\n")


def main():
    set_seed()
    feature_npy = f"dataset/{DATA}_feature_{DATASET_NUM}.npy"
    label_npy = f"dataset/{DATA}_label_{DATASET_NUM}.npy"
    # load
    X = np.load(feature_npy, allow_pickle=True)
    y = np.load(label_npy, allow_pickle=True)
    X_train, X_test, y_train, y_test = setup_dataset(X, y)
    env = HighUtilityItemsetsMining()
    models = train(X_train, y_train)
    test(X_test, y_test, models)
    test_sample(X_test, y_test, models, 10)
    test_each_length(X_test, y_test, models)

    for i, model in enumerate(models):
        path = f"model/lgb/{DATA}_{DATASET_NUM}_{i}.txt"
        model.save_model(path)


def predict():
    set_seed()
    feature_npy = f"dataset/{DATA}_feature_{DATASET_NUM}.npy"
    label_npy = f"dataset/{DATA}_label_{DATASET_NUM}.npy"
    # load
    X = np.load(feature_npy, allow_pickle=True)
    y = np.load(label_npy, allow_pickle=True)
    X_train, X_test, y_train, y_test = setup_dataset(X, y)
    env = HighUtilityItemsetsMining()

    models = []
    for i in range(5):
        path = f"model/lgb/{DATA}_{DATASET_NUM}_{i}.pth"
        models.append(lgb.Booster(model_file=path))

    test(X_test, y_test, models)
    test_each_length(X_test, y_test, models)


if __name__ == "__main__":
    main()
