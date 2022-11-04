import numpy as np
from typing import List


def save_list_np_array(list_to_save: List[np.ndarray], filename: str):
    to_save = np.array(list_to_save, dtype=object)
    np.save(filename, to_save)


def load_list_np_array(filename: str) -> List:
    all_data_numpy = np.load(filename, allow_pickle=True)
    return [constraint for constraint in all_data_numpy]
