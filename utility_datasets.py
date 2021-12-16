import numpy as np
from hui_env import HighUtilityItemsetsMining
from config import DATASET_NUM, DATA


if __name__ == "__main__":
    data_path = f"data/{DATA}_spmf.txt"
    output_feature_npy = f"dataset/{DATA}_feature_{DATASET_NUM}.npy"
    output_label_npy = f"dataset/{DATA}_label_{DATASET_NUM}.npy"
    # DATASET_NUM個のデータセットを作る
    x, y = [], []
    # ランダムにbit_vectorを生成し、そのUtilityを計算する
    env = HighUtilityItemsetsMining(data_path=data_path)
    for i in range(DATASET_NUM):
        if i % (DATASET_NUM // 1000) == 0:
            print(f"{i} dataset create!")
        # bit_vector, utility = env.valid_sample()
        bit_vector, utility = env.uniform_valid_sample(i)
        x.append(list(bit_vector))
        y.append(utility)
    # ファイルへ保存
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    np.save(output_feature_npy, x)
    print(f"save complete npy: {output_feature_npy} {x.shape}")
    np.save(output_label_npy, y)
    print(f"save complete npy: {output_label_npy} {y.shape}")
