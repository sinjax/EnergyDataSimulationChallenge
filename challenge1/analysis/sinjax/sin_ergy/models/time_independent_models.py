import numpy as np
np.random.seed(1)

from keras.callbacks import EarlyStopping
from keras.engine import Input
from keras.models import Model
from keras.layers import Dense, merge
from keras.optimizers import Adam
from sklearn.cross_validation import train_test_split

from data.utils import load_training, load_test


class LinearEnergy(Model):
    def __init__(self, input_size):
        input = Input(shape=(input_size,))
        dense = Dense(1)

        super().__init__(input=input, output=dense(input))

class DeeperEnergy(Model):
    def __init__(self, input_size):
        input = Input(shape=(input_size,))
        output = Dense(100,activation='relu')(input)
        output = Dense(10,activation='relu')(output)
        output = Dense(100,activation='relu')(output)

        output = merge([input,output],mode='concat')
        output = Dense(1)(output)

        super().__init__(input=input, output=output)

ALL_COLS = ["House", "Year", "Month", "Temperature", "Daylight", "PreviousEnergyProduction"]
FEWER_COLS = ["Year", "Month", "Temperature", "Daylight", "PreviousEnergyProduction"]
JUST_PREVIOUS = ["PreviousEnergyProduction"]
NO_YEAR = ["Month", "Temperature", "Daylight", "PreviousEnergyProduction"]

cols = NO_YEAR
(X,Y) = load_training(cols=cols)
X_test, y_test = load_test(cols=cols)

X_train, X_val, y_train, y_val = train_test_split(X,Y,test_size=0.2,random_state=1)


# model = LinearEnergy(len(cols))
model = DeeperEnergy(len(cols))
model.compile(optimizer="rmsprop",loss='mape')
model.fit(X_train,y_train,
          validation_data=(X_val,y_val),
          nb_epoch=1000,batch_size=32,
          callbacks=[EarlyStopping(patience=1)]

)

print(model.evaluate(X_val,y_val,verbose=0))
print(model.evaluate(X_test,y_test,verbose=0))