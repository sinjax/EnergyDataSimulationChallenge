import os
import pprint
import random
from collections import defaultdict

import numpy as np

from sin_ergy.models import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import model_from_yaml
from sklearn.cross_validation import train_test_split

from sin_ergy.data.utils import load_training, load_test, prepare_training_cv, load_normal, load_sequential, \
    training_cv_splits

import keras.backend as K

ROOT = "/Users/ss/Development/python/EnergyDataSimulationChallenge/challenge1/analysis/sinjax"
MODELS_ROOT = "%s/models"%ROOT
if not os.path.exists(MODELS_ROOT): os.makedirs(MODELS_ROOT)
ALL_COLS = ["House", "Year", "Month", "Temperature", "Daylight", "PreviousEnergyProduction"]
FEWER_COLS = ["Year", "Month", "Temperature", "Daylight", "PreviousEnergyProduction"]
JUST_PREVIOUS = ["PreviousEnergyProduction"]
NO_YEAR = ["Month", "Temperature", "Daylight", "PreviousEnergyProduction"]
NO_TIME = ["Temperature", "Daylight", "PreviousEnergyProduction"]
cols = NO_TIME

def model_location(name,fold):
    return os.sep.join([MODELS_ROOT, name, str(fold)])

def callbacks(name, model, fold):
    model_root = model_location(name, fold)
    if not os.path.exists(model_root): os.makedirs(model_root)
    weights_out = os.sep.join([model_root,"model.h5"])
    schema_out = os.sep.join([model_root,"model.schema"])
    with open(schema_out,"w") as sfile:
        sfile.write(model.to_yaml())
    return [
        EarlyStopping(patience=2),
        ModelCheckpoint(weights_out,save_best_only="True")
    ]
def model_exists(name, fold):
    model_root = model_location(name, fold)
    return os.path.exists(model_root)

def load_model(name, fold):
    model_root = model_location(name, fold)
    weights_out = os.sep.join([model_root,"model.h5"])
    schema_out = os.sep.join([model_root,"model.schema"])
    with open(schema_out,"r") as sfile:
        model = model_from_yaml(sfile.read(), globals())

    model.load_weights(weights_out)
    return model
if __name__ == "__main__":
    random.seed(1)
    prepare_training_cv()

    models = {
        LinearEnergy,
        DeeperEnergy
    }
    sequence_models = {
        # SequenceEnergyWithDense,
        ConvEnergy,
        DeepConvEnergy
    }
    results = defaultdict(list)
    np.random.seed(1)
    for model_creator in models.union(sequence_models):
        name = model_creator.__name__
        print("Starting to train %s"%name)
        for fold, (training_dataframe, val_dataframe, test_dataframe) in enumerate(training_cv_splits(cols)):
            if model_creator in models:
                data_loader = lambda df: load_normal(df, cols)
                model_loader = lambda X: model_creator(X.shape[1])
            elif model_creator in sequence_models:
                data_loader = lambda df: load_sequential(df, cols, window_size=12, only_final_y=True)
                model_loader = lambda X: model_creator(X.shape[1],X.shape[2])
            else:
                raise Exception("can't train this model")

            X_train, y_train = data_loader(training_dataframe)
            X_val, y_val = data_loader(val_dataframe)
            X_test, y_test = data_loader(test_dataframe)
            print("train: %s"%str(X_train.shape))
            print("test: %s"%str(X_test.shape))
            print("val: %s"%str(X_val.shape))

            if not model_exists(name, fold):
                model = model_loader(X_train)
                model.compile(optimizer="adam",loss='mape')

                cb = callbacks(name, model, fold)
                model.fit(
                    X_train,y_train,
                    validation_data=(X_val,y_val),
                    nb_epoch=1000,batch_size=32,
                    callbacks=cb
                )
            model = load_model(name, fold)
            model.compile(optimizer="adam", loss='mape')

            eval_score = model.evaluate(X_test, y_test, verbose=0)
            print("Fold %d, Model: %s, evaluation: %s"%(fold, name, eval_score))
            results[model_creator.__name__] += [eval_score]
    results = dict(
        [(x,np.mean(y)) for (x,y) in results.items()]
    )
    pp = pprint.PrettyPrinter()
    pp.pprint(results)