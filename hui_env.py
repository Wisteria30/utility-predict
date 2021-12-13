# -*- coding: utf-8 -*-

from collections import namedtuple
from functools import lru_cache
import numpy as np


LIMIT = 100000


def convert_list_str2float32(str_list):
    return list(map(np.float32, str_list.split()))


def load_hui_db(path):
    with open(path) as f:
        transaction = f.readlines()

    # Database: Dictionary type transaction list of itemsets and utilities
    database = np.empty(0, dtype=np.float32)
    # items: Set of all items
    items = set()
    # utils: List of utility values for each transaction
    utils = np.empty(0, dtype=np.float32)

    for t in transaction:
        left, center, right = t.split(":")
        left = convert_list_str2float32(left)
        right = convert_list_str2float32(right)
        database = np.append(database, {left[i]: right[i] for i in range(len(left))})
        items = items | set(left)
        utils = np.append(utils, np.float32(center))

    return database, items, utils


class HighUtilityItemsetsMining:
    def __init__(
        self,
        data_path="data/chess_utility_spmf.txt",
        cache_limit=100000,
    ):
        self.database, self.items, self.utils = load_hui_db(data_path)
        # Specify the maximum LRU cache as a global variable
        global LIMIT
        LIMIT = cache_limit
        self._setup_db()

    def _setup_db(self):
        self.b2i_dict = {i: v for i, v in enumerate(self.items)}
        self.i2b_dict = {v: i for i, v in enumerate(self.items)}

        self.bit_map = np.zeros((len(self.database), len(self.items)), dtype=np.int)
        for i, transaction in enumerate(self.database):
            for item in transaction:
                self.bit_map[i][self.i2b_dict[item]] = self.database[i][
                    item
                ].astype(np.int)

    def _create_random_pbv(self):
        return np.random.binomial(1, 0.5, len(self.items))

    def calc_utility_bv(self, bv):
        itemsets = self.convert_bv2tuple_x(bv)
        print("this bit-vector represented itemsets is ")
        print(itemsets)
        return self._calc_utility(tuple(bv))

    def calc_utility_x(self, x):
        bv = convert_tuple_x2bv(x)
        return self._calc_utility(tuple(bv))

    def convert_bv2tuple_x(self, bv):
        return tuple(sorted([int(self.b2i_dict[i]) for i, v in enumerate(bv) if v == 1]))

    def convert_tuple_x2bv(self, x):
        bv = np.array([0 for _ in range(len(self.items))], dtype=np.int8)
        for item in x:
            bv[self.i2b_dict[item]] = 1
        return bv

    @lru_cache(maxsize=LIMIT)
    def _calc_utility(self, bv):
        bv = np.array(bv, dtype=np.int8)

        bv_mask = bv > 0
        # After masking the bit-vector, check if all the elements of the itemset are present.
        # masking the bit-vector
        filtered = self.bit_map[:, bv_mask]
        if filtered.size == 0:
            return 0
        # Extract columns where all elements of the itemset are zero or greater and calculate the sum.
        utility = np.sum(filtered[np.all(filtered, axis=1)])
        return utility
    
    # create Random Itemset and Reward(this itemset)
    def sample(self):
        bv = self._create_random_pbv()
        utility = self._calc_utility(tuple(bv))
        # bv = bv.astype(np.float32)
        return bv, utility

    # create Random and valid Itemset and Reward(this itemset)
    def valid_sample(self):
        # sampleing transaction
        random_t = np.random.randint(self.bit_map.shape[0])
        sample_transaction = self.bit_map[random_t]
        # Randomly create an itemset from only the items that appear in the transaction.
        prob_bv = [0.5 if i > 0 else 0 for i in sample_transaction]
        bv = np.random.binomial(1, prob_bv)
        utility = self._calc_utility(tuple(bv))
        # bv = bv.astype(np.float32)
        return bv, utility
