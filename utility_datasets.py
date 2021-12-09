import numpy as np
from hui_env import HighUtilityItemsetsMining


if __name__ == "__main__":
    DATASET_NUM = 1000000
    data = "chess_utility"
    data_path = f"data/{data}_spmf.txt"
    output_npy = f"dataset/{data}_{DATASET_NUM}.npy"
    # DATASET_NUM個のデータセットを作る
    x, y = [], []
    # ランダムにbit_vectorを生成し、そのUtilityを計算する
    env = HighUtilityItemsetsMining()
    for _ in range(DATASET_NUM):
        bit_vector, utility = env.sample()
        x.append(bit_vector)
        y.append(utility)
    # ファイルへ保存
    np.save(output_npy, (x, y))
    print(f"npyを保存しました: {output_npy} {len(x)}")

