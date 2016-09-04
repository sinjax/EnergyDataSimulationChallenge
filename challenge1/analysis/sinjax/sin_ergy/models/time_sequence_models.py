from keras.engine import Input
from keras.layers import Dense, merge, LSTM, Masking, Lambda, Flatten, Dropout
from keras.models import Model


class SequenceEnergy(Model):
    def __init__(self, ndays=None ,input_size=None,input=None, output=None, name=None):
        if input and output:
            super().__init__(input=input, output=output, name=name)
            return
        input = Input(shape=(ndays, input_size))
        output = Masking()(input)
        output = LSTM(32,activation='linear')(output)
        output = Dense(1)(output)

        super().__init__(input=input, output=output)

class SequenceEnergyWithDense(Model):
    def __init__(self, ndays=None,input_size=None,input=None, output=None, name=None):
        if input and output:
            super().__init__(input=input, output=output, name=name)
            return
        input = Input(shape=(ndays, input_size))
        output = Masking()(input)
        output = LSTM(32,activation='linear')(output)
        output = Dense(100)(output)
        output = Dense(10)(output)
        output = Dense(100)(output)
        output = Dense(1)(output)

        super().__init__(input=input, output=output)

class SequenceEnergyWithDenseAndInput(Model):
    def __init__(self, ndays=None,input_size=None,input=None, output=None, name=None):
        if input and output:
            super().__init__(input=input, output=output, name=name)
            return
        input = Input(shape=(ndays, input_size))
        output = Masking()(input)
        output = LSTM(100,activation="linear")(output)
        output = Dense(100)(output)
        output = Dense(20)(output)
        output = Dense(4,activation="relu")(output)
        final_input = Lambda(
            lambda x: x[:,-1,:],
            output_shape=lambda x_shape: (x_shape[0],x_shape[2])
        )(input)
        output = merge([final_input,output],mode='concat')
        output = Dense(1)(output)

        super().__init__(input=input, output=output)

class SequenceEnergyFlatAndInput(Model):
    def __init__(self, ndays=None,input_size=None,input=None, output=None, name=None):
        if input and output:
            super().__init__(input=input, output=output, name=name)
            return
        input = output = Input(shape=(ndays, input_size))
        output = LSTM(32,activation="linear",return_sequences=True)(output)
        output = Flatten()(output)
        output = Dense(4,activation="relu")(output)
        output = Dropout(0.2)(output)
        final_input = Lambda(
            lambda x: x[:,-1,:],
            output_shape=lambda x_shape: (x_shape[0],x_shape[2])
        )(input)
        output = merge([final_input,output],mode='concat')
        output = Dense(1)(output)

        super().__init__(input=input, output=output)