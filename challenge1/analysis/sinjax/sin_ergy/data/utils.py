import os
import random

import numpy as np
import pandas
from ..settings import *
import csv

def _load_data(d):
    d = pandas.read_csv(d)
    d["PreviousEnergyProduction"] = d.EnergyProduction \
        .shift(1) \
        .fillna(d.EnergyProduction.mean())
    return d

def load_training(cols=None, sequence=False, window_size=None):
    if cols is None:
        cols = ["House", "Year", "Month", "Temperature", "Daylight"]
    d = _load_data(os.sep.join([DATAROOT, TRAINING]))
    if not sequence:
        return d[cols].as_matrix(), d["EnergyProduction"].as_matrix()
    else:
        X = []
        Y = []
        for x in d.groupby("House"):
            X_toadd = x[1][cols].as_matrix()
            Y_toadd = np.array([x[1]["EnergyProduction"].as_matrix()]).T
            if window_size is None or window_size > X_toadd.shape[0]:
                X += [X_toadd]
                Y += [Y_toadd]
            else:
                for start in range((X_toadd.shape[0] - window_size) + 1):
                    X += [X_toadd[start:start+window_size,:]]
                    Y += [Y_toadd[start:start+window_size,:]]

        return np.array(X), np.array(Y)


def load_test(cols=None, sequence=False, window_size=None):
    if cols is None:
        cols = ["House", "Year", "Month", "Temperature", "Daylight"]
    d = _load_data(os.sep.join([DATAROOT, ALL]))

    if not sequence:
        return d[cols].as_matrix()[23::24, :], np.array([d["EnergyProduction"].as_matrix()[23::24]]).T
    else:
        X = []
        Y = []
        for x in d.groupby("House"):
            X_toadd = x[1][cols].as_matrix()
            Y_toadd = np.array([x[1]["EnergyProduction"].as_matrix()]).T
            if window_size is None or window_size > X_toadd.shape[0]:
                X += [X_toadd]
                Y += [Y_toadd]
            else:
                start = X_toadd.shape[0] - window_size
                X += [X_toadd[start:start+window_size,:]]
                Y += [Y_toadd[start:start+window_size,:]]
        return np.array(X), np.array(Y)


def load_all(cols=None, sequence=False, window_size=None):
    if cols is None:
        cols = ["House", "Year", "Month", "Temperature", "Daylight"]
    d = _load_data(os.sep.join([DATAROOT, ALL]))
    if not sequence:
        return d[cols].as_matrix(), d["EnergyProduction"].as_matrix()
    else:
        X = []
        Y = []
        for x in d.groupby("House"):
            X_toadd = x[1][cols].as_matrix()
            Y_toadd = np.array([x[1]["EnergyProduction"].as_matrix()]).T
            if window_size is None or window_size > X_toadd.shape[0]:
                X += [X_toadd]
                Y += [Y_toadd]
            else:
                for start in range((X_toadd.shape[0] - window_size) + 1):
                    X += [X_toadd[start:start+window_size,:]]
                    Y += [Y_toadd[start:start+window_size,:]]
        return np.array(X), np.array(Y)

def load_normal(data_frame, cols):
    return data_frame[cols].as_matrix(), data_frame["EnergyProduction"].as_matrix()

def load_sequential(data_frame, cols, window_size=12, only_final_y=True):
    X = []
    Y = []
    for x in data_frame.groupby("House"):
        X_toadd = x[1][cols].as_matrix()
        Y_toadd = np.array([x[1]["EnergyProduction"].as_matrix()]).T
        if window_size is None or window_size > X_toadd.shape[0]:
            X += [X_toadd]
            Y += [Y_toadd]
        else:
            for start in range((X_toadd.shape[0] - window_size) + 1):
                X += [X_toadd[start:start+window_size,:]]
                Y += [Y_toadd[start:start+window_size,:]]
    if only_final_y:
        return np.array(X), np.array(Y)[:,-1,:]
    else:
        return np.array(X), np.array(Y)

def prepare_training_cv():
    fold_root_dir = os.sep.join([DATAROOT,CV])
    if os.path.exists(fold_root_dir): return
    os.makedirs(fold_root_dir)
    n_folds = int(N_FOLDS)
    fold_csv = {}
    all = ["ID","Label","House","Year","Month","Temperature","Daylight","EnergyProduction"]
    for fold in range(n_folds):
        fold_root = os.sep.join([fold_root_dir,str(fold)])
        os.makedirs(fold_root)
        writer = open(os.sep.join([fold_root, "data.csv"]), "w")
        writer.write(",".join(all) + "\n")
        fold_csv[fold] = writer


    d = _load_data(os.sep.join([DATAROOT, TRAINING]))
    for house, rows in d.groupby("House"):
        fold_csv[random.randint(0,n_folds-1)].write(
            rows.to_csv(header=False)
        )

    for fold, writer in fold_csv.items():
        writer.close()

def training_cv_splits(cols):
    fold_root_dir = os.sep.join([DATAROOT,CV])

    n_folds = int(N_FOLDS)
    for fold in range(n_folds):

        test_fold = fold
        val_fold = ( fold + 1 ) % n_folds
        training_folds = set(range(n_folds)) - {test_fold} - {val_fold}

        test_df = _load_single_fold(test_fold, fold_root_dir)
        val_df = _load_single_fold(val_fold, fold_root_dir)
        training_df = pandas.concat([_load_single_fold(f, fold_root_dir) for f in training_folds])

        yield training_df, val_df, test_df


def _load_single_fold(fold, fold_root):
    return _load_data(os.sep.join([fold_root, str(fold), "data.csv"]))