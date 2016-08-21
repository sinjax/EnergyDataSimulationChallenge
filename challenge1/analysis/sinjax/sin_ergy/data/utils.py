import os

import pandas

DATAROOT = "/Users/ss/Development/python/EnergyDataSimulationChallenge/challenge1/sin_ergy.data"
ALL = "dataset_500.csv"
TRAINING = "training_Dataset_500.csv"
TEST = "test_dataset_500.csv"


def _load_data(d,cols):
    d = pandas.read_csv(d)
    d["PreviousEnergyProduction"] = d.EnergyProduction.shift(1).fillna(d.EnergyProduction.mean())
    X = d.as_matrix(columns=cols)
    Y = d.as_matrix(columns=["EnergyProduction"])
    return X,Y

def load_training(cols=None):
    if cols is None:
        cols = ["House", "Year", "Month", "Temperature", "Daylight"]
    return _load_data(os.sep.join([DATAROOT, TRAINING]),cols)

def load_test(cols=None):
    if cols is None:
        cols = ["House", "Year", "Month", "Temperature", "Daylight"]
    X,y = _load_data(os.sep.join([DATAROOT, ALL]),cols)
    return X[23::24,:], y[23::24,:]

def load_all(cols=None):
    if cols is None:
        cols = ["House", "Year", "Month", "Temperature", "Daylight"]
    return _load_data(os.sep.join([DATAROOT, ALL]),cols)

