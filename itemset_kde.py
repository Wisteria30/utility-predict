import numpy as np
import torch

# KDE
from scipy.stats import gaussian_kde
import matplotlib
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from itertools import combinations

from config import set_seed, DATASET_NUM, DATA
from hui_env import HighUtilityItemsetsMining


def generate_n_itemsets(item_len, n):
    item_len_c_n = 1
    for i in range(n):
        item_len_c_n *= (item_len - i)
    item_len_c_n //= n
    itemsets = np.zeros([item_len_c_n, item_len])
    for i, v in enumerate(combinations(range(item_len), n)):
        for j in v:
            itemsets[i][j] = 1
    return itemsets


def kde(data_list, s):
    plt.style.use('ggplot')
    # x軸をGridするためのデータも生成
    x_grid = np.linspace(min(data_list), max(data_list), num=len(data_list))
    # データを正規化したヒストグラムを表示する用
    weights = np.ones_like(data_list)/float(len(data_list))

    kde_model = gaussian_kde(data_list)
    y = kde_model(x_grid)

    fig = plt.figure(figsize=(14,7))
    plt.plot(x_grid, y)
    plt.hist(data_list, alpha=0.3, bins=20, weights=weights)

    fig.savefig(f"kde-{s}.png")
    print("graph save!")


def bar(data_list, s):
    x = np.array(range(1, len(data_list) + 1))

    fig = plt.figure(figsize=(14,7))
    plt.bar(x, data_list)
    # plt.plot(x, data_list)

    fig.savefig(f"bar-{s}.png")
    print("graph save!")


def log_log_graph(data_list, s):
    x = np.array(range(1, len(data_list) + 1))

    fig = plt.figure(figsize=(14,7))
    plt.xscale('log')
    plt.yscale('log')

    plt.xlim(1, len(data_list) + 1)
    plt.ylim(1, data_list[0])
    plt.xlabel('Rank')
    plt.ylabel('Frequent')
    plt.scatter(x, data_list)

    fig.savefig(f"log-log-{s}.png")
    print("graph save!")


def plogp():
    bins = 1000
    x = np.array([i / bins for i in range(1, 1 * bins)])
    y = x * np.log(x)

    fig = plt.figure(figsize=(14,7))
    plt.plot(x, y)

    fig.savefig(f"plogp.png")
    print("graph save!")


def kde_itemsets():
    set_seed()

    env = HighUtilityItemsetsMining()
    # 長さ1のやつでテスト
    ones = np.array([1] * len(env.items), dtype=np.float32)
    X = np.diag(ones)
    n = 2
    for i in range(2, n + 1):
        itemsets = generate_n_itemsets(len(env.items), i)
        X = np.concatenate([X, itemsets])

    # utilities = np.array([env.calc_utility_bv(bv)[1] for bv in X])
    # kde(utilities, "utility")
    frequent = np.array([env.calc_frequent_bv(bv)[1] for bv in X])
    frequent = np.sort(frequent)[::-1]
    # kde(frequent, "frequent")
    # bar(frequent, "frequent")
    log_log_graph(frequent, "frequent")


if __name__ == "__main__":
    plogp()
    # kde_itemsets()
