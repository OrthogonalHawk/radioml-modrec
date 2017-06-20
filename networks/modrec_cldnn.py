#!/usr/bin/python
#!/bin/env python

from copy import deepcopy
import cPickle
import logging
import math
import numpy
from keras.callbacks import TensorBoard
from keras.models import Model
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers import concatenate
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import Reshape
from keras.layers.recurrent import LSTM
from keras.layers import MaxPooling1D
from keras.layers import RepeatVector
from random import shuffle
from sklearn.metrics import mean_squared_error
import sys

NUM_CNN_OUTPUTS = 50
CNN_KERNEL = (1, 8)

NUM_LSTM_OUTPUTS = 64

DROPOUT_PCT = 0.5

'''
    Defines a Deep Neural Network (DNN) composed of Convolutional and Long Short
     Term Memory layers, a CLDNN, as described in: https://arxiv.org/pdf/1703.09197.pdf
'''

class modrec_cldnn(object):

    def __init__(self, _input_shape):
        """
          Arguments:
            _input_shape: a tuple in the form [batch_size, channels, height, width]

          For IQ or FFT inputs, the height is set to 1 and the width is the length
           of the IQ vector or the number of FFT bins. The channels value is then
           set to 2, which accounts for the real and imaginary input components.

          Note that the returned model does NOT include an output layer, softmax
           or otherwise. It is left to the caller to decide how to process the output.
        """

        logging.info("Creating a CLDNN model...")

        self.input_shape = _input_shape[1:]

        # create the input layer
        self.input = Input(shape=self.input_shape, dtype='float32', name='cldnn_input')


        # create the first Convolutional layer and add dropout
        self.cnn_01_out = Conv2D(NUM_CNN_OUTPUTS,
                                 CNN_KERNEL,
                                 strides=(1,1),
                                 padding='same',
                                 data_format='channels_first',
                                 activation='relu',
                                 use_bias=True,
                                 name='cldnn_conv1')(self.input)
        self.cnn_01_out = Dropout(DROPOUT_PCT, name='cldnn_conv1_drp')(self.cnn_01_out)


        # create the second Convolutional layer and add dropout
        self.cnn_02_out = Conv2D(NUM_CNN_OUTPUTS,
                                 CNN_KERNEL,
                                 strides=(1,1),
                                 padding='same',
                                 data_format='channels_first',
                                 activation='relu',
                                 use_bias=True,
                                 name='cldnn_conv2')(self.cnn_01_out)
        self.cnn_02_out = Dropout(DROPOUT_PCT, name='cldnn_conv2_drp')(self.cnn_02_out)


        # create the third Convolutional layer and add dropout
        self.cnn_03_out = Conv2D(NUM_CNN_OUTPUTS,
                                 CNN_KERNEL,
                                 strides=(1,1),
                                 padding='same',
                                 data_format='channels_first',
                                 activation='relu',
                                 use_bias=True,
                                 name='cldnn_conv3')(self.cnn_02_out)
        self.cnn_03_out = Dropout(DROPOUT_PCT, name='cldnn_conv3_drp')(self.cnn_03_out)


        # concatenate the output from the first Convolutional layer with the
        #  output of the third layer.
        self.cnn_out = concatenate([self.cnn_01_out, self.cnn_03_out])


        # reshape the output so that it is compatible with the LSTM layer(s). basically
        #  throwing out the dummy height layer here...
        self.cnn_out_reshape = Reshape((1024, 100))(self.cnn_out)


        # create the first LSTM layer
        self.lstm_01_out = LSTM(NUM_LSTM_OUTPUTS,
                                dropout=DROPOUT_PCT,
                                return_sequences=True,
                                name='cldnn_lstm1')(self.cnn_out_reshape)


        # create the second LSTM layer
        self.lstm_02_out = LSTM(NUM_LSTM_OUTPUTS,
                                dropout=DROPOUT_PCT,
                                return_sequences=False,
                                name='cldnn_lstm2')(self.lstm_01_out)


        # create a 'dummy' generic output so that there is an easy input/output
        #  naming convention for a caller
        self.output = self.lstm_02_out


        # create a model from the defined layers
        self.model = Model(self.input, self.output)

        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            self.model.summary()
