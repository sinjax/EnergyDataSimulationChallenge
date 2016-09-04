from keras.engine import Input
from keras.layers import Dense, merge, LSTM, Masking, Lambda, Flatten, Dropout, Conv1D, MaxPooling1D, AveragePooling1D, \
    GlobalMaxPooling1D, GlobalAveragePooling1D, TimeDistributed
from keras.models import Model


class ConvEnergy(Model):
    def __init__(self, ndays=None,input_size=None,input=None, output=None, name=None):
        if input and output:
            super().__init__(input=input, output=output, name=name)
            return
        output = input = Input(shape=(ndays, input_size))
        output = Conv1D(5,3,input_shape=(ndays,input_size))(output)
        output = Flatten()(output)
        output = Dense(1)(output)

        super().__init__(input=input, output=output)

class DeepConvEnergy(Model):
    def __init__(self, ndays=None,input_size=None,input=None, output=None, name=None, nfilters=32, filterwidth=3):
        if input and output:
            super().__init__(input=input, output=output, name=name)
            return
        output = input = Input(shape=(ndays, input_size))
        output = Conv1D(nfilters,filterwidth,input_shape=(ndays,input_size))(output)
        doutput = output = Flatten()(output)
        # doutput = Dense(100,activation='relu')(doutput)
        doutput = Dense(10,activation='sigmoid')(doutput)
        output = merge([output,doutput],mode='concat')
        output = Dense(1)(output)

        super().__init__(input=input, output=output)

