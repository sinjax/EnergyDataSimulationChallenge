from keras.engine import Input
from keras.models import Model
from keras.layers import Dense, merge


class LinearEnergy(Model):
    def __init__(self, input_size=None,input=None, output=None, name=None):
        if input and output:
            super().__init__(input=input, output=output, name=name)
            return
        input = Input(shape=(input_size,))
        dense = Dense(1)

        super().__init__(input=input, output=dense(input))

class DeeperEnergy(Model):
    def __init__(self, input_size=None,input=None, output=None, name=None):
        if input and output:
            super().__init__(input=input, output=output, name=name)
            return
        input = Input(shape=(input_size,))
        output = Dense(100,activation='relu')(input)
        output = Dense(10,activation='relu')(output)
        output = Dense(100,activation='relu')(output)

        output = merge([input,output],mode='concat')
        output = Dense(1)(output)

        super().__init__(input=input, output=output)