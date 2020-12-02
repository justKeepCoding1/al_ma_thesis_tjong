import numpy as np
import random
import torch
from scipy.stats import entropy
from concurrent import futures
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import time

from al_kitti.presets import segmen_preset as segmen_preset


def parallel_label_comp(label_num, output_list):

    label_stack = np.ones_like(output_list) * label_num
    v_yi_c_label = np.sum(output_list == label_stack, axis=0)/output_list.shape[0]

    return v_yi_c_label


def vote_entropy_semseg(output_list, labels_valid):
    # output_list: n_models x n_data x H x W
    # v_yi_c.shape: n_data x n_classes x H x W
    # output_list = output_list.argmax(axis=2).astype(np.int8)  # n_models x n_data x H x W

    # prepare parallel vote entropy calculation
    num_cores = multiprocessing.cpu_count()
    v_yi_c_list = []
    with ProcessPoolExecutor(max_workers=1) as executor:
        v_yi_c_all = [executor.submit(parallel_label_comp, x, output_list) for x in list(range(labels_valid.__len__()))]
        for v_yi_c_label in futures.as_completed(v_yi_c_all):
            v_yi_c_list.append(v_yi_c_label.result())
    v_yi_c = np.array(v_yi_c_list)
    # should be: n_data x H x W, then sum for every pixel
    en = entropy(np.array(v_yi_c), base=2, axis=0)  # vote entropy for each pixel in each data
    # average ve on every pixel, shape: length of batch
    result = np.sum(en, axis=(-2, -1))/(output_list.shape[-2]*output_list.shape[-1])
    return result


# using proper numpy implementation, this one is faster
def vote_entropy_semseg_old(output_list, labels_valid):
    # output_list: n_models x n_data x H x W
    # label_stack_list: n_classes x n_models x n_data x H x W

    label_stack_list = []
    for i_label in range(labels_valid.__len__()):
        label_stack = np.ones_like(output_list) * i_label
        label_stack_list.append(label_stack)

    # v_yi_c: n_classes x n_data x H x W
    # expanded_output: n_classes x n_models x n_data x H x W, should have sma shape with label_stack_list
    expanded_output = np.concatenate([np.expand_dims(output_list, 0)] * labels_valid.__len__()).astype(np.int8)
    v_yi_c = (np.sum(expanded_output == np.array(label_stack_list), axis=1).astype(np.float32) / output_list.shape[0])

    # en: n_data x H x W, then sum for every pixel
    en = entropy(v_yi_c, base=2, axis=0)  # vote entropy for each pixel in each data
    # average ve on every pixel, shape: length of batch
    result = np.sum(en, axis=(-2, -1))/(output_list.shape[-2]*output_list.shape[-1])
    return result


def get_next_indices(idx_library, ve_train_all, idx_ratio, batch_train_size, data_train_len):

    indices = np.array([]).astype(int)
    indices_en = np.argsort((-1) * ve_train_all)
    indices_en = indices_en[np.isin(indices_en, np.array(idx_library), invert=True)]  # exclude used data

    # append from highest entropy
    n_highest_en = int(idx_ratio[0] * batch_train_size)
    from_highest_en = indices_en[0:n_highest_en]  # indices_en[0:0] returns empty
    indices = np.append(indices, from_highest_en)
    # print("Entropy: " + str(from_highest_en.__len__()))

    # add random index
    random_suggest = np.array(
        random.sample(range(data_train_len), k=(batch_train_size + idx_library.__len__())))   # create random index
    random_suggest = random_suggest[
        np.isin(random_suggest, np.append(idx_library, indices), invert=True)]  # exclude index already included
    random_suggest = random_suggest[0:int(batch_train_size * idx_ratio[1])]
    indices = np.append(indices, random_suggest)  # append to training index
    # print("Random: " + str(random_suggest.__len__()))

    idx_library = np.append(idx_library, indices)
    # print(idx_library)

    return idx_library
