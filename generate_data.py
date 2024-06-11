#!/usr/bin/env python
# coding: utf-8

import pickle
import torch
import numpy as np
import random
import argparse
from tqdm import tqdm
from pypots.utils.random import set_random_seed

def main(test_path, train_path, val_path, output_dir):
    set_random_seed()

    # Load data
    with open(test_path, 'rb') as f:
        data_test = pickle.load(f)
    with open(train_path, 'rb') as f:
        data_train = pickle.load(f)
    with open(val_path, 'rb') as f:
        data_val = pickle.load(f)

    # Convert data
    ped_test = np.array(data_test['obs_traj'].permute(0, 2, 1))
    ped_train = np.array(data_train['obs_traj'].permute(0, 2, 1))
    ped_val = np.array(data_val['obs_traj'].permute(0, 2, 1))

    final_train = ped_train
    final_test = ped_test
    final_val = ped_val

    final_test_conc = np.concatenate((ped_test, ped_test, ped_test, ped_test, ped_test))

    # Introduce missing values in training data
    missing_values_list = []
    X_org_train = final_train.copy()
    total_values = 8
    list1 = [0, 1, 2, 3, 4]
    for traj in X_org_train:
        missing_values_count = random.choice(list1)
        missing_indices = np.random.choice(total_values, missing_values_count, replace=False)
        traj2 = traj
        traj2[missing_indices] = np.nan
        missing_values_list.append(traj2)
    X_intact_train = np.array(missing_values_list)
    train_mask = np.isnan(X_intact_train)
    print("Number of missing values introduced in training data:", np.isnan(X_intact_train).sum())

    # Introduce missing values in validation data
    val_missing_values_list = []
    X_org_val = final_val.copy()
    for val_traj in X_org_val:
        missing_values_count = random.choice(list1)
        val_missing_indices = np.random.choice(total_values, missing_values_count, replace=False)
        val_traj2 = val_traj
        val_traj2[val_missing_indices] = np.nan
        val_missing_values_list.append(val_traj2)
    X_intact_val = np.array(val_missing_values_list)
    val_mask = np.isnan(X_intact_val)
    print("Number of missing values introduced in val data:", np.isnan(X_intact_val).sum())

    # Introduce missing values in test data
    test_missing_values_list = []
    X_org_test = final_test.copy()
    for item in list1:
        for test_traj in X_org_test:
            items = item
            missing_values_count = items
            test_missing_indices = np.random.choice(total_values, missing_values_count, replace=False)
            test_traj2 = np.copy(test_traj)  # Create a copy of test_traj
            test_traj2[missing_indices] = np.nan
            test_missing_values_list.append(test_traj2)
    X_intact_test = np.array(test_missing_values_list)
    test_mask = np.isnan(X_intact_test)
    print("Number of missing values introduced in test data:", np.isnan(X_intact_test).sum())

    # Update test data with masks and trajectories
    data_test['missing_mask'] = test_mask
    data_test['obs_traj'] = X_intact_test
    data_test['pred_traj'] = torch.cat([data_test['pred_traj']] * 5)

    final_test_past = data_test['obs_traj']
    final_test_future = data_test['pred_traj'].permute(0, 2, 1)

    test_accel = np.concatenate((final_test_past, final_test_future), axis=1)
    test_flattened_arr = test_accel.reshape(-1, 2)
    test_first_diff = np.diff(test_flattened_arr, axis=0)
    zero_row = np.zeros((1, 2))
    test_first_diff = np.concatenate((zero_row, test_first_diff), axis=0)
    test_first_diff = test_first_diff.reshape(-1, 20, 2)
    test_first_diff = np.float32(test_first_diff)
    test_first_diff = test_first_diff[:, :8, :]

    data_test['obs_traj_rel'] = test_first_diff
    data_test['pred_traj_rel'] = data_test['pred_traj_rel'].permute(0, 2, 1)
    data_test['pred_traj'] = final_test_future

    data_test['pred_traj_rel'] = torch.cat([data_test['pred_traj_rel']] * 5)
    data_test['loss_mask'] = torch.cat([data_test['loss_mask']] * 5)
    data_test['non_linear_ped'] = torch.cat([data_test['non_linear_ped']] * 5)
    data_test['seq_start_end'] = data_test['seq_start_end'] * 5

    # Update train data with masks and trajectories
    data_train['missing_mask'] = train_mask
    data_train['obs_traj'] = X_intact_train

    final_train_past = data_train['obs_traj']
    final_train_future = data_train['pred_traj'].permute(0, 2, 1)

    train_accel = np.concatenate((final_train_past, final_train_future), axis=1)
    train_flattened_arr = train_accel.reshape(-1, 2)
    train_first_diff = np.diff(train_flattened_arr, axis=0)
    zero_row = np.zeros((1, 2))
    train_first_diff = np.concatenate((zero_row, train_first_diff), axis=0)
    train_first_diff = train_first_diff.reshape(-1, 20, 2)
    train_first_diff = np.float32(train_first_diff)
    train_first_diff = train_first_diff[:, :8, :]
    data_train['obs_traj_rel'] = train_first_diff

    data_train['pred_traj_rel'] = data_train['pred_traj_rel'].permute(0, 2, 1)
    data_train['pred_traj'] = final_train_future

    # Update validation data with masks and trajectories
    data_val['missing_mask'] = val_mask
    data_val['obs_traj'] = X_intact_val

    final_val_past = data_val['obs_traj']
    final_val_future = data_val['pred_traj'].permute(0, 2, 1)

    val_accel = np.concatenate((final_val_past, final_val_future), axis=1)
    val_flattened_arr = val_accel.reshape(-1, 2)
    val_first_diff = np.diff(val_flattened_arr, axis=0)
    zero_row = np.zeros((1, 2))
    val_first_diff = np.concatenate((zero_row, val_first_diff), axis=0)
    val_first_diff = val_first_diff.reshape(-1, 20, 2)
    val_first_diff = np.float32(val_first_diff)
    val_first_diff = val_first_diff[:, :8, :]
    data_val['obs_traj_rel'] = val_first_diff

    data_val['pred_traj_rel'] = data_val['pred_traj_rel'].permute(0, 2, 1)
    data_val['pred_traj'] = final_val_future

    # Save data
    with open(f'{output_dir}/data_test.pkl', 'wb') as f:
        pickle.dump(data_test, f)

    with open(f'{output_dir}/data_train.pkl', 'wb') as f:
        pickle.dump(data_train, f)

    with open(f'{output_dir}/data_val.pkl', 'wb') as f:
        pickle.dump(data_val, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process trajectory data and introduce missing values.")
    parser.add_argument('test_path', type=str, help="Path to the test dataset file")
    parser.add_argument('train_path', type=str, help="Path to the train dataset file")
    parser.add_argument('val_path', type=str, help="Path to the validation dataset file")
    parser.add_argument('output_dir', type=str, help="Directory to save the processed files")
    args = parser.parse_args()

    main(args.test_path, args.train_path, args.val_path, args.output_dir)
