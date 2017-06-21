#!/usr/bin/python
#!/bin/env python

import argparse
import cPickle
import logging
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.models import Model
from keras.layers import Dense
from sklearn.metrics import mean_squared_error
from random import shuffle
import sys

import modrec_cldnn


# define constants for the command-line argument parser
DESCRIPTION = "Trains and evaluates a CLDNN Autoencoder model using the Keras framework"
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_NUM_TRAINING_EPOCHS = 10


def get_one_hot_output_vector(output_classes, cur_class):

    ret = np.zeros([len(output_classes)], dtype=np.uint8)
    ret[output_classes.index(cur_class)] = 1

    return ret


def get_dataset_vectors(dataset):
    data_x, data_y, num_classes = [], [], 0

    # get the set of output_classes
    output_classes = []
    for key in dataset.keys():
        output_classes.append(key[0])
    output_classes = sorted(list(set(output_classes)))
    num_classes = len(output_classes)


    # get the data_x and data_y lists
    for key in dataset:
        output_vec = get_one_hot_output_vector(output_classes, key[0])
        for idx in range(dataset[key].shape[0]):
            data_x.append(np.expand_dims(dataset[key][idx], 1))
            data_y.append(output_vec)

    # now shuffle the lists
    data_x_shuf, data_y_shuf = [], []
    idx_shuf = range(len(data_x))
    shuffle(idx_shuf)
    for idx in idx_shuf:
        data_x_shuf.append(data_x[idx])
        data_y_shuf.append(data_y[idx])

    return np.array(data_x_shuf), np.array(data_y_shuf), num_classes


def evaluate_predictions(predict_y, true_y, threshold):
    correct, missed, false_positives = 0, 0, 0

    for idx in range(len(predict_y)):
        for true_idx in range(len(true_y[idx])):
            if true_y[idx][true_idx] == 1:
                if predict_y[idx][true_idx] >= threshold:
                    correct += 1
                else:
                    missed += 1

            else:
                if predict_y[idx][true_idx] >= threshold:
                    false_positives += 1

    return correct, missed, false_positives


def train_cldnn(args):

    # load the dataset and shuffle
    logging.info("Loading dataset from %s..." % (args.input_data_file))
    dataset = cPickle.load( open(args.input_data_file, 'rb') )

    data_x, data_y, NUM_OUTPUT_CLASSES = get_dataset_vectors(dataset)

    if args.load_model == None:
        # get the modrec_cldnn model
        cldnn = modrec_cldnn.modrec_cldnn((-1, 2, 1, 1024))

        # add an output layer
        model_output = Dense(NUM_OUTPUT_CLASSES,
                             activation='sigmoid',
                             name='cldnn_output')(cldnn.output)

        # create a model with the desired output
        cldnn_model = Model(cldnn.input, model_output)

        logging.info("Compiling the CLDNN model...")
        cldnn_model.compile(loss='mean_squared_error',
                            optimizer='adam',
                            metrics=['mean_squared_error'])

    else:

        logging.info("Loading pre-trained model from %s" % (args.load_model))
        cldnn_model = load_model(args.load_model)


    if not args.evaluate:
        logging.info("Starting model training...")
        cldnn_model.fit(data_x, data_y,
                        epochs=args.num_epochs,
                        batch_size=16,
                        verbose=2,
                        shuffle=True,
                        validation_split=0.25,
                        callbacks=[TensorBoard(log_dir='/tmp/cldnn', histogram_freq=0, write_graph=False),
                                   ModelCheckpoint('/tmp/cldnn/saved_cldnn_model.hdf5', monitor='mean_squared_error', verbose=1, save_best_only=True, mode='min')])

    logging.info("Evaluating the CLDNN model...")
    num_validation_examples = int(len(data_x) * 0.25)
    predict_y = cldnn_model.predict(data_x[-num_validation_examples:])
    correct, missed, false_positives = evaluate_predictions(predict_y,
                                                            data_y[-num_validation_examples:],
                                                            args.confidence_threshold)

    logging.info("Evaluated against %d examples. Correct=%u, Missed=%u, FalsePositives=%u" % \
        (num_validation_examples, correct, missed, false_positives))


if __name__ == "__main__":

    # create a command-line argument parser
    parser = argparse.ArgumentParser(description=DESCRIPTION)

    # optional argument(s)
    parser.add_argument('--confidence_threshold', default=DEFAULT_CONFIDENCE_THRESHOLD, type=float, \
                        help='Required confidence threshold; defaults to %f' % (DEFAULT_CONFIDENCE_THRESHOLD))

    parser.add_argument('--evaluate', action='store_true', \
                        help='Evaluate a pre-trained model, skip training')

    parser.add_argument('--load_model', default=None, type=str, \
                        help='Load a pre-trained model from the provided path')

    parser.add_argument('--num_epochs', default=DEFAULT_NUM_TRAINING_EPOCHS, type=int, \
                        help="Number of epochs for training")

    parser.add_argument('--verbose', action='store_true', \
                        help='Increase logging verbosity')


    # required argument(s)
    parser.add_argument('input_data_file', \
                        help="Dataset to load; assuming that it contains a pickled data structure")


    # parse the command-line arguments
    args = parser.parse_args()

    # setup the logging utility
    active_log_level = logging.INFO
    if args.verbose:
        active_log_level = logging.DEBUG
    logging.basicConfig(level=active_log_level,
                        format='%(asctime)s [CLDNN] [%(levelname)s] %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')

    train_cldnn(args)
